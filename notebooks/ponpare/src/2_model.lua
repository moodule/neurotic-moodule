----------------------------------------------------------------------
-- Ponpare contest on Kaggle
--
-- This script defines the ML model
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'math'

-----------------------------------------------------------------------

if not opt then
  print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Ponpare Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-trainset',     'detail',        'the dataset for the training : detail | detail_50 | visits')
   cmd:option('-model',        'mlp_28x14x2',   'model to be trained and/or evaluated')
   cmd:text()
   opt = cmd:parse(arg or {})
end

cachePath = paths.concat('cache', opt.trainset)
resultsPath = paths.concat('results', opt.trainset, opt.model)

-----------------------------------------------------------------------
print '==> constructing model'

-- model parameters
nInput = trainData.data:size(2)
nHidden1 = trainData.data:size(2)
nHidden2 = 21
nHidden3 = 14
nOutput = 2

-- the actual model : sequential multi layer perceptron
model = nn.Sequential()
model:add(nn.Linear(nInput, nHidden1))
model:add(nn.Tanh())
model:add(nn.Linear(nHidden1, nHidden2))
model:add(nn.Tanh())
model:add(nn.Linear(nHidden2, nHidden3))
model:add(nn.Tanh())
model:add(nn.Linear(nHidden3, nOutput))
model:add(nn.LogSoftMax())     -- LogSoftMax

-----------------------------------------------------------------------
print '==> defining a procedure to load the models'

function load_model()
  -- location to the file containing the serialized model object
  modelFile = paths.concat(resultsPath, 'model.net')
  
  print ('[model] loading '..modelFile)
    
  -- read the serialized object on the disk
  model = torch.load(modelFile)

  -- read the serialized mean, standard deviation and feature's names
  features = torch.load(paths.concat(cachePath, 'features.t7'))
  mean = torch.load(paths.concat(cachePath, 'mean.t7'))
  stdv = torch.load(paths.concat(cachePath, 'stdv.t7'))

  -- set to evaluate mode
  model:evaluate()
end
