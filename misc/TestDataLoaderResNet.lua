require 'hdf5'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local t = require 'misc.transforms'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
	
	-- load the json file which contains additional information about the dataset
	print('DataLoader loading json file: ', opt.json_file)
	self.info = utils.read_json(opt.json_file)
	self.itow = self.info.itow
	self.vocab_size = utils.count_keys(self.itow)

	self.batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
	print('vocab size is ' .. self.vocab_size)
	
	-- open the hdf5 file
	print('DataLoader loading h5 file: ', opt.h5_file)
	self.h5_file = hdf5.open(opt.h5_file, 'r')
 
	-- extract image size from dataset
	local images_size = self.h5_file:read('/images'):dataspaceSize()
	assert(#images_size == 4, '/images should be a 4D tensor')
	assert(images_size[3] == images_size[4], 'width and height must match')
	self.num_images = images_size[1]
	self.num_channels = images_size[2]
	self.max_image_size = images_size[3]

	print(string.format('read %d images of size %dx%dx%d', self.num_images, 
						self.num_channels, self.max_image_size, self.max_image_size))

	self.split_ix = {}
	self.iterator = {}
	local splits = self.h5_file:read("/splits"):all()
	for i = 1, splits:size(1) do
		local split
		if splits[i] == 1 or splits[i] == 3 then
			split = "train"
		elseif splits[i] == 2 then
			split = "val"
		elseif splits[i] == 4 then
			split = "test"
		end
		if not self.split_ix[split] then
			-- initialize new split
			self.split_ix[split] = {}
			self.iterator[split] = 1
		end
		table.insert(self.split_ix[split], i)
	end

	self.__size = {}
	for k,v in pairs(self.split_ix) do
		print(string.format('assigned %d images to split %s', #v, k))
	end

	self.meanstd = {
				mean = { 0.485, 0.456, 0.406 },
				std = { 0.229, 0.224, 0.225 },
			}

	self.transform = t.Compose{
		 t.ColorNormalize(self.meanstd)
	}
end

function DataLoader:init_rand(split)
	local size = #self.split_ix[split]	
	if split == 'train' then
		self.perm = torch.randperm(size)
	else
		self.perm = torch.range(1,size) -- for test and validation, do not permutate
	end
end

function DataLoader:reset_iterator(split)
	self.iterator[split] = 1
end

function DataLoader:getVocabSize()
	return self.vocab_size
end

function DataLoader:getVocab()
	return self.itow
end

function DataLoader:getnBatch(split)
	return math.ceil(#self.split_ix[split] / self.batch_size)
end

function DataLoader:run(split)
	local size, batch_size = #self.split_ix[split], self.batch_size
	local num_channels, max_image_size = self.num_channels, self.max_image_size

	local split_ix = self.split_ix[split]
	local idx = self.iterator[split]
	
	if idx <= size then
		batch_size = math.min(batch_size, size - idx + 1)
		local indices = self.perm:narrow(1, idx, batch_size)

		local img_batch_raw = torch.ByteTensor(batch_size, 3, 256, 256)
		local img_id_batch = torch.LongTensor(batch_size):zero()
		for i, ixm in ipairs(indices:totable()) do
							
			local ix = split_ix[ixm]
			img_batch_raw[i] = self.h5_file:read("images"):partial({ix, ix}, {1, num_channels}, {1, max_image_size}, {1, max_image_size})
			
			img_id_batch[i] = self.h5_file:read("/imageids"):partial({ix, ix})
		end

		local data_augment = false
		if split == 'train' then
			data_augment = true
		end

		local h,w = img_batch_raw:size(3), img_batch_raw:size(4)
		local cnn_input_size = 224
		-- cropping data augmentation, if needed
		if h > cnn_input_size or w > cnn_input_size then 
			local xoff, yoff
			if data_augment then
				xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
			else
				-- sample the center
				xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
			end
			-- crop.
			img_batch_raw = img_batch_raw[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1}}]
		end
		
		img_batch_raw = self.transform(img_batch_raw:float():div(255))
		--img_batch_raw = img_batch_raw:float():div(255)

		local batch_data = {}
		batch_data.images = img_batch_raw
		batch_data.img_id = img_id_batch

		self.iterator[split] = self.iterator[split] + batch_size
		return batch_data
	end
end

