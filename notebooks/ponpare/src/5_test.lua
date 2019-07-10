----------------------------------------------------------------------
-- Ponpare contest on Kaggle
--
-- This script implements a test procedure, to evaluate a model.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'float' then input = input:float()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   local class_accuracy = confusion.totalValid * 100
   local purchase_recall = (confusion.mat[2][2] / confusion.mat[2]:sum()) * 100
   local purchase_accuracy = (confusion.mat[2][2] / confusion.mat[{ {}, {2} }]:sum()) * 100
   testLogger:add{['% accuracy (test)'] = class_accuracy}     --, ['% purchase accuracy (test)'] = purchase_accuracy}
   if opt.plot then
      testLogger:style{['% accuracy (test)'] = '-', ['% purchase accuracy (test)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end
