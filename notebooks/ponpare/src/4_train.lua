----------------------------------------------------------------------
-- Ponpare contest on Kaggle
--
-- This script :
--   + constructs mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Ponpare Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-visualize',     false,      'visualize input data and weights during training')
   cmd:option('-plot',          false,      'live plot')
   cmd:option('-optimization',  'SGD',      'optimization method: SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate',  1e-3,       'learning rate at t=0')
   cmd:option('-l1',            0.,         'regularization parameter l1 wrt norm 1')
   cmd:option('-l2',            0.,         'regularization parameter l2 wrt norm 2')
   cmd:option('-batchSize',     1,          'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay',   0,          'weight decay (SGD only)')
   cmd:option('-momentum',      0,          'momentum (SGD only)')
   cmd:option('-t0',            1,          'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter',       2,          'maximum nb of iterations for CG and LBFGS')
   cmd:option('-model',        'mlp_28x14x2',  'model to be trained and/or evaluated')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------

if opt.type == 'cuda' then
  print '==> moving everything to the GPU'
   model:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

print(opt)

-- some globals
nSamples = trainData:size()
classes = {'not purchased','purchased'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(resultsPath, 'train.log'))
testLogger = optim.Logger(paths.concat(resultsPath, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = nSamples * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(nSamples)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,nSamples,opt.batchSize do
      -- disp progress
      xlua.progress(t, nSamples)

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,nSamples) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         if opt.type == 'double' then input = input:double()
         elseif opt.type == 'cuda' then input = input:cuda() end
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
        -- get new parameters
        if x ~= parameters then
          parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        -- f is the average of all criterions
        local f = 0

        -- evaluate function for complete mini batch
        for i = 1,#inputs do
          -- estimate f
          local output = model:forward(inputs[i])
          local err = criterion:forward(output, targets[i])
          
          f = f + err
          
          -- estimate df/dW
          local df_do = criterion:backward(output, targets[i])
          model:backward(inputs[i], df_do)
          
          -- regularization penalties (L1 and L2):
          if opt.l1 ~= 0 or opt.l2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.l1 * norm(parameters,1)
            f = f + opt.l2 * 0.5 * norm(parameters,2)^2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.l1) + parameters:clone():mul(opt.l2) )
          end

          -- update confusion
          confusion:add(output, targets[i])
        end

        -- normalize gradients and f(X)
        gradParameters:div(#inputs)
        f = f/#inputs

        -- return f and df/dX
        return f,gradParameters
      end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / nSamples
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   local class_accuracy = confusion.totalValid * 100
   local purchase_recall = (confusion.mat[2][2] / confusion.mat[2]:sum()) * 100
   local purchase_accuracy = (confusion.mat[2][2] / confusion.mat[{ {}, {2} }]:sum()) * 100
   trainLogger:add{['% accuracy (train)'] = class_accuracy}    --, ['% purchase accuracy (train)'] = purchase_accuracy}
   if opt.plot then
      trainLogger:style{['% accuracy (train)'] = '-', ['% purchase accuracy (train)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(resultsPath, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
