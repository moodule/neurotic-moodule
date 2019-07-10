----------------------------------------------------------------------
-- Ponpare contest on Kaggle
-- 
-- This script loads and preprocesses the data
----------------------------------------------------------------------

require 'torch'             -- torch
require 'nn'                -- provides a normalization operator
require 'csvigo'            -- tool to load csv files
require 'paths'             -- to manipulate files
require '1_datafunctions'   -- custom lib

----------------------------------------------------------------------

if not opt then
  print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Ponpare Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-trainset',     'detail',        'the dataset for the training : detail | detail_50 | visits')
   cmd:option('-validset',     'detail',        'the dataset for the cross validation : detail | detail_50 | visits')
   cmd:option('-testset',      'detail',        'the dataset for the testing : detail | detail_50 | visits')
   cmd:option('-model',        'mlp_28x14x2',   'model to be trained and/or evaluated')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> initializing the environment'

-- create the directories to store the computation's (intermediate) results
dataPath = paths.concat('..', 'data', opt.trainset)
postPath = paths.concat('cache', 'post')
cachePath = paths.concat('cache', opt.trainset)
resultsPath = paths.concat('results', opt.trainset, opt.model)
os.execute('mkdir -p '..cachePath)
os.execute('mkdir -p '..postPath)
os.execute('mkdir -p '..resultsPath)

-- define some globals
trainData = {}
validData = {}
testData = {}
features = {}   -- the names of the features
classes = {'not purchased','purchased'}   -- the names of the classes
mean = {}   -- mean, calculated on the training set
stdv = {}   -- standard deviation, calculated on the training set

----------------------------------------------------------------------
print '==> loading the datasets'

local xTrainCsvFile = paths.concat(dataPath, 'x_train.csv')
local yTrainCsvFile = paths.concat(dataPath, 'y_train.csv')

local xTrainTorchFile = paths.concat(cachePath, 'x_train.t7')
local yTrainTorchFile = paths.concat(cachePath, 'y_train.t7')

local xValidTorchFile = paths.concat(cachePath, 'x_valid.t7')
local yValidTorchFile = paths.concat(cachePath, 'y_valid.t7')

local xTestTorchFile = paths.concat(cachePath, 'x_test.t7')
local yTestTorchFile = paths.concat(cachePath, 'y_test.t7')

if (not paths.filep(xTrainTorchFile)) or (not paths.filep(yTrainTorchFile)) then
  
  -- read the csv files (the header line is ignored)
  local dataCsv = csvigo.load{path=xTrainCsvFile, mode='tidy'}
  local labelsCsv = csvigo.load{path=yTrainCsvFile, mode='tidy'}
  
  -- get the names of the features and classes (that respectively label the input and the output)
  features = get_keys(dataCsv)
  labelsKeys = get_keys(labelsCsv)
  
  -- get the dimensions of the set
  local nSamples = #dataCsv[features[1]]
  local nFeatures = #features
  
  -- create the tensors
  local dataTensor = torch.Tensor(nSamples, nFeatures)
  local labelsTensor = torch.Tensor(nSamples)
  
  -- put the data in the corresponding tensor
  csv_to_tensor(dataCsv, dataTensor, features)
  csv_to_tensor(labelsCsv, labelsTensor, labelsKeys)
  
  -- normalize the data
  labelsTensor:add(1)     -- so that it contains the class label : 0 => 1 (not purchased) | 1 => 2 (purchased)
  normalize_tensor(dataTensor, mean, stdv)
  
  -- split the tensors
  local dataTrain, labelsTrain, dataTemp, labelsTemp = split_dataset(dataTensor, labelsTensor, 0.7)
  local dataValid, labelsValid, dataTest, labelsTest = split_dataset(dataTemp, labelsTemp, 0.0)
  
  -- fill the train table
  trainData['data'] = dataTrain
  trainData['labels'] = labelsTrain
  function trainData:size() return trainData.data:size(1) end
  
  -- fill the validation table
  validData['data'] = dataValid
  validData['labels'] = labelsValid
  function validData:size() return validData.data:size(1) end
  
  -- fill the test table
  testData['data'] = dataTest
  testData['labels'] = labelsTest
  function testData:size() return testData.data:size(1) end
  
  -- save the tensors in torch format for future (faster) use
  torch.save(xTrainTorchFile, dataTrain)
  torch.save(yTrainTorchFile, labelsTrain)
  torch.save(xValidTorchFile, dataValid)
  torch.save(yValidTorchFile, labelsValid)
  torch.save(xTestTorchFile, dataTest)
  torch.save(yTestTorchFile, labelsTest)
  torch.save(paths.concat(cachePath, 'features.t7'), features)
  torch.save(paths.concat(cachePath, 'classes.t7'), classes)
  torch.save(paths.concat(cachePath, 'mean.t7'), mean)
  torch.save(paths.concat(cachePath, 'stdv.t7'), stdv)
  
else
  
  -- assume ALL the variables extracted from the dataset have been previously dumped
  
  -- load the previously saved tensors
  local dataTrain = torch.load(xTrainTorchFile)
  local labelsTrain = torch.load(yTrainTorchFile)
  local dataValid = torch.load(xValidTorchFile)
  local labelsValid = torch.load(yValidTorchFile)
  local dataTest = torch.load(xTestTorchFile)
  local labelsTest = torch.load(yTestTorchFile)
  features = torch.load(paths.concat(cachePath, 'features.t7'))
  classes = torch.load(paths.concat(cachePath, 'classes.t7'))
  mean = torch.load(paths.concat(cachePath, 'mean.t7'))
  stdv = torch.load(paths.concat(cachePath, 'stdv.t7'))
  
  -- fill the train table
  trainData['data'] = dataTrain
  trainData['labels'] = labelsTrain
  function trainData:size() return trainData.data:size(1) end
  
  -- fill the validation table
  validData['data'] = dataValid
  validData['labels'] = labelsValid
  function validData:size() return validData.data:size(1) end
  
  -- fill the test table
  testData['data'] = dataTest
  testData['labels'] = labelsTest
  function testData:size() return testData.data:size(1) end
  
end
