----------------------------------------------------------------------
-- Ponpare contest on Kaggle
-- 
-- This script puts everything together :
-- 1) load and preprocess the data
-- 2) setup the ML Model
-- 3) define the loss criterion and convert the label tensor
-- 4) define the training algorithm and optimize the model
-- 5) test and xvalidation
-- 6) process and visualize the results !
----------------------------------------------------------------------

require 'torch'

-- TODO ignore indices corresponding to the coupon ids
-- TODO convertir les ids utilisateurs en nombre
-- TODO faire une prédiction sur les données finales kaggle
-- TODO remettre en forme pour les poster
-- TODO améliorer les fonctionnalités de test : faire une dizaine de 
-- découpe du jeu en train/test et moyenner les performances
-- TODO enregistrer l'évolution des performances au cours de l'entrainement
-- TODO séparer le dataset en deux seulement : 60%

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Ponpare Purchasing Predictions')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-mode',         'pred',      'action to perform : train | predict')
cmd:option('-seed',          2308,        'fixed input seed for repeatable experiments')
cmd:option('-type',         'float',      'type: double | float | cuda')
cmd:option('-threads',       1,           'number of threads')
-- data:
cmd:option('-trainset',     'detail',     'the dataset for the training : detail | detail_50 | visits')
cmd:option('-validset',     'detail',     'the dataset for the cross validation : detail | detail_50 | visits')
cmd:option('-testset',      'detail',     'the dataset for the testing : detail | detail_50 | visits')
-- criterion
cmd:option('-weight',        1e-4,        'weight of the class "not puchased" compared to "purchased"')
cmd:option('-loss',         'nll',        'type of loss function to minimize: nll | mse | margin')
-- training:
cmd:option('-plot',          true,        'live plot')
cmd:option('-optimization', 'SGD',        'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate',  1e-3,        'learning rate at t=0')
cmd:option('-l1',            0.,          'regularization parameter l1 wrt norm 1')
cmd:option('-l2',            2e-5,        'regularization parameter l2 wrt norm 2')
cmd:option('-batchSize',     4,           'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay',   5e-5,        'weight decay (SGD only)')
cmd:option('-momentum',      0,           'momentum (SGD only)')
cmd:option('-t0',            1,           'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter',       2,           'maximum nb of iterations for CG and LBFGS')
cmd:option('-model',        'mlp_28x28x21x14x2',    'subdirectory to save/log experiments in')
-- postprocessing:
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all !'

dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_test.lua'
dofile '6_post.lua'


if opt.mode == 'train' then
  
  ----------------------------------------------------------------------
  print '==> training!'

  while true do
     train()
     test()
  end
  
else
  
  ----------------------------------------------------------------------
  print '==> predicting!'
  
  load_model()
  
  for i=0,15 do
    load_dataset(i)
    predict(i)
  end
  
end
