s = torch.LongTensor{5,6}
m = nn.Sequential()
p = nn.Parallel(2,2)
for i = 1,2 do
    p:add(nn.LookupTable(s[i], 3))
end
m:add(p)
m:add(nn.Linear(6,3))
m:add(nn.LogSoftMax())
for i = 1,1000 do
    x = torch.Tensor(3,1)
    x:random(5)
    t = torch.Tensor(3,1)
    t:random(6)
    x = torch.cat(x,t)
    y = torch.Tensor(3)
    y:random(3)
    pred = m:forward(x)
    loss = nn.ClassNLLCriterion()
    local err = loss:forward(pred, y)
    local gradLoss = loss:backward(pred, y)
    m:zeroGradParameters()
    m:backward(x, gradLoss)
    m:updateParameters(0.01)
    print(err)
end
