require 'torch'
require 'nn'
require 'math'

----------------------------------------------------------------------
-- TABLE HANDLING
----------------------------------------------------------------------

function transpose_table(t)
  local transposedTable = {}
  
  for i,v in ipairs(t) do
    transposedTable[v] = i
  end
  
  return transposedTable
    
end

----------------------------------------------------------------------
-- CSV HANDLING
----------------------------------------------------------------------

-- get the table of keys of a given table
function get_keys(t)
  local keys = {}
  local i = 1
  
  for k,v in pairs(t) do
    keys[i] = k
    i = i + 1
  end
  
  return keys
end

-- transfer data from a csv table to a tensor
function csv_to_tensor(csvTable, tensor, features, destination, startingIndice)
  local nFeatures = #features
  local nSamples = #csvTable[features[1]]
  local indices = destination or transpose_table(features)
  local start = startingIndice or 1
  
  for j,f in ipairs(features) do
    for i=1,nSamples do
      if nFeatures == 1 then
        tensor[i] = tonumber(csvTable[f][i])
      else
        if f == 'USER_ID_hash' then
          tensor[i][1] = tonumber(csvTable[f][i])
        elseif f == 'COUPON_ID_hash' then
          tensor[i][2] = tonumber(csvTable[f][i])
        else
          local col = start + indices[f] - 1
          tensor[i][col] = tonumber(csvTable[f][i])
        end
      end
    end
  end
end

----------------------------------------------------------------------
-- DATASET PREPROCESSING
----------------------------------------------------------------------

-- shuffle and split a tensor to extract training and testing samples for a dataset
function split_dataset(data, labels, ratio)
   --local shuffle = torch.randperm(data:size(1))     -- no more permutation : it is done during the training
   local numTrain = math.floor(data:size(1) * ratio)
   local numTest = data:size(1) - numTrain
   local trainData = torch.Tensor(numTrain, data:size(2))
   local testData = torch.Tensor(numTest, data:size(2))   
   local trainLabels = torch.Tensor(numTrain)
   local testLabels = torch.Tensor(numTest)
   
   for i=1,numTrain do
      trainData[i] = data[i]:clone()
      trainLabels[i] = labels[i]       -- labels[k] is a number, not a tensor (since labels has only one dimension)
   end
   for i=numTrain+1,numTrain+numTest do
      testData[i-numTrain] = data[i]:clone()
      testLabels[i-numTrain] = labels[i]   -- same as before : labels is a number and NOT a tensor
   end
   return trainData, trainLabels, testData, testLabels
end

-- normalize all the features of the data and return the mean and standard deviation as tables
function normalize_tensor(tensor, meanTable, stdvTable)
  local nSamples = tensor:size(1)
  local nFeatures = tensor:size(2)
  
  for j=1,nFeatures do
    
    -- compute the mean for the jth feature
    meanTable[j] = tensor[{{},j}]:mean()
    -- substract the mean to all samples, on the jth feature
    tensor[{{},j}]:add(-meanTable[j])
  
    -- compute the standard deviation for the jth feature
    stdvTable[j] = tensor[{{},j}]:std()
    -- divide all the samples by the standard deviation, on the jth feature
    tensor[{{},j}]:div(stdvTable[j])
      
  end
  
end

-- convert label ids to vectors (0, 1, 0, ..., 0)
function labels_to_vector(labels, nClass)
    local nSamples = labels:size(1)
    local vectorLabels = torch.Tensor(nSamples, nClass):fill(-1)
    
    -- put ones in the column corresponding to each label's class
    for i = 1,nSamples do
       vectorLabels[{ i, labels[i] }] = 1
    end
    
    return vectorLabels
end