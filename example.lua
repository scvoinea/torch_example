x = torch.Tensor(3,2)             -- batch of three examples, each a 1D vector of length 2.
x[{{},1}]:random(5)               -- First feature categorical with 5 levels.
x[{{},2}]:random(6)               -- Second feature categorical with 6 levels.

y = torch.Tensor(3)
y:random(3)

m = nn.Sequential()

p = nn.Parallel(2,2)              -- Want one lookup table for each feature.
p:add(nn.LookupTable(5, 3))
p:add(nn.LookupTable(6, 3))

m:add(p)
m:add(nn.Linear(6,3))
m:add(nn.LogSoftMax())
    
pred = m:forward(x)
loss = nn.ClassNLLCriterion()
err = loss:forward(pred, y)
gradLoss = loss:backward(pred, y)  -- ERROR: /Users/voinea/torch/install/share/lua/5.1/nn/Parallel.lua:54: inconsistent tensor size at /Users/voinea/torch/pkg/torch/lib/TH/generic/THTensorCopy.c:7
m:zeroGradParameters()
m:backward(x, gradLoss)
m:updateParameters(0.01)
