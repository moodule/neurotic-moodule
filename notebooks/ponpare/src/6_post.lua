----------------------------------------------------------------------
-- Ponpare contest on Kaggle
--
-- This script converts the dataset back to its original format, to
-- be posted on kaggle.
--
-- 0) forward les données kaggle et obtenir les prédictions sur les 
-- couples coupon / utilisateur
-- 1) créer une table:
--  => key = user id (nb)
--  => val = table des coupons id (nb)
-- 2) ordonner la table par probabilité croissantes des prédictions
-- 3) convertir les id en string
-- 4) enregistrer au format csv
----------------------------------------------------------------------

require 'torch'             -- torch
require 'optim'             -- an optimization package, for model evaluation
require 'csvigo'            -- tool to load csv files
require 'paths'             -- to manipulate files
require '1_datafunctions'   -- custom lib

----------------------------------------------------------------------

if not opt then
  print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Ponpare Dataset Postprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-type',      'float',          'type: double | float | cuda')
   cmd:option('-model',     'mlp_28_14_2',    'name of the model to use for prediction')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> loading the model'

-- set the default tensor type
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end


----------------------------------------------------------------------
print '==> defining the dataset loading procedure'

function load_dataset(id)
  
  if (not paths.filep(paths.concat(postPath, 'x_'..id..'.t7'))) then
    
    ----------------------------------------------------------------------
    print('['..id..'] reading the csv file')
    
    local postCsv = csvigo.load{path=paths.concat('..', 'data', 'post', 'x_'..id..'.csv'), mode='tidy'}
    local keys = get_keys(postCsv)
      
    -- get the dimensions of the posting set
    local nSamples = #postCsv[keys[1]]
    local nKeys = #keys

    -- create the tensors
    local postTensor = torch.Tensor(nSamples, nKeys)

    -- populate the tensor
    csv_to_tensor(postCsv, postTensor, keys, transpose_table(features), 3)      -- the values are set in the same column indices as the tensor used for the training
  
    ----------------------------------------------------------------------
    print('['..id..'] normalizing the tensor')

    for j=3,nKeys do
      -- substract the mean to all samples, on the jth feature
      postTensor[{{},j}]:add(-mean[j-2])

      -- divide all the samples by the standard deviation, on the jth feature
      postTensor[{{},j}]:div(stdv[j-2])
        
    end
    
    ----------------------------------------------------------------------
    print('['..id..'] serializing to torch binary')
    
    -- serialize the dataset to load it faster next time
    torch.save(paths.concat(postPath, 'x_'..id..'.t7'), postTensor)
    
  end
  
end

----------------------------------------------------------------------
print('==> creating the prediction procedure')

function predict(id)
  
  ----------------------------------------------------------------------
  print('['..id..'] loading the subset')
  
  -- create the prediction table
  local output = {}
  output['USER_ID'] = {}
  output['COUPON_ID'] = {}
  output['LIKELIHOOD'] = {}

  -- load the serialized file
  local postTensor = torch.load(paths.concat(postPath, 'x_'..id..'.t7'))
  local nSamples = postTensor:size(1)
  local nKeys = postTensor:size(2)
  
  ----------------------------------------------------------------------
  print('['..id..'] predicting purchased coupons')
  
  for t = 1,nSamples do
    
    -- disp progress
    xlua.progress(t, nSamples)

    -- get new sample
    local input = postTensor[{ {t}, {3,nKeys} }]

    -- make the prediction
    local prediction = model:forward(input)
    prediction:exp()

    -- save prediction
    output['USER_ID'][t] = postTensor[t][1]
    output['COUPON_ID'][t] = postTensor[t][2]
    output['LIKELIHOOD'][t] = prediction[1][2]
    
  end
  
  ----------------------------------------------------------------------
  print('['..id..'] saving the predictions')
  csvigo.save{path=paths.concat(resultsPath, 'predictions_'..id..'.csv'), data=output}
  
end

