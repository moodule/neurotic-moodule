----------------------------------------------------------------------
-- Ponpare contest on Kaggle
--
-- This script :
--  + define the loss function  = negative-log likelihood,
--    using log-normalized output units (SoftMax)
--  + transforms the label ids to vectors
----------------------------------------------------------------------

require 'torch'             -- torch
require 'nn'                -- provides all sorts of loss functions

dofile '1_datafunctions.lua'

----------------------------------------------------------------------
-- parse command line arguments

if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Ponpare Loss Function')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-loss',          'nll',        'type of loss function to minimize: nll | mse | margin')
   cmd:option('-weight',         0.1,         'weight of the class "not puchased" compared to "purchased"')
   cmd:text()
   opt = cmd:parse(arg or {})

   -- to enable self-contained execution:
   model = nn.Sequential()
end

----------------------------------------------------------------------
print '==> define loss'

-- number of classes
nOutput = 2

-- weight of the classes (relative importance of the prediction for each class)
local weightTensor = torch.Tensor(2)
weightTensor[1] = opt.weight             -- not purchased
weightTensor[2] = 1.0 - opt.weight       -- purchased

-- classification criterion
if opt.loss == 'mse' then
  criterion = nn.WeightedMSECriterion(torch.Tensor({3,1}))
else
  criterion = nn.ClassNLLCriterion(torch.Tensor({3,1}))
end

-- convert the labels ids to vectors, if required by the criterion
if trainData and opt.loss == 'mse' then
  ----------------------------------------------------------------------
  print '==> convert the label ids to vectors (0, 0, 1, 0, ...)'
  
  -- convert training labels
  trainData.labels = labels_to_vector(trainData.labels, nOutput)
  
  -- convert validation labels
  --validData.labels = labels_to_vector(validData.labels, nOutput)

  -- convert validation labels
  testData.labels = labels_to_vector(testData.labels, nOutput)
end
