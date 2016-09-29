#!/bin/env julia

# My ANN module
using CANN
# The FANN julia module, interfacing with C FANN
using FANN


## Create all possible 8-bit strings, shuffled
eight_bits = zeros(256,8);
for i in 0:255
	eight_bits[i+1,:]=float(split(bits(int8(i)),""));
end
i = [1:256];
shuffle!(i);
eight_bits = eight_bits[i,:]

cycle_one = zeros(8,8);
for i in 1:8
	cycle_one[i,i] = 1.
end

## Training and predicting with FANN
dset = DataSet(eight_bits', eight_bits')
net = ANN([8, 3, 8], [:sigmoid_symmetric, :sigmoid_symmetric]; b=0.1, errorfunc=:tanh, steepness=9.0)
@time train!(net, dset, max_epochs=80000, epochs_between_reports=100, desired_error=1e-8)

fann_error = 0. 
for i in 1:size(eight_bits)[1]
	fann_error += mean_squared_error(predict(net, vec(eight_bits[i,:])), vec(eight_bits[i,:]))
end


## Traning and predicting with CANN
targets = features = eight_bits;
nnet = NNet(targets, features, [8, 3, 8]; cutoff=0.0, g=tanh, g′=sech, h=tanh);
@time nnet.learn!()

cann_error = 0.
for i in 1:size(eight_bits)[1]
	cann_error += mean_squared_error(nnet.predict(vec(eight_bits[i,:])), vec(eight_bits[i,:]))
end



## Traning and predicting with CANN
targets = features = cycle_one;
nnet = NNet(targets, features, [8, 3, 8]; cutoff=0.0, α=3.0);
@time nnet.learn!()

for i in 1:size(cycle_one)[1]
	ppvec(nnet.predict(vec(cycle_one[i,:])))
end


## Traning and predicting with CANN
nnet = NNet(targets, features, [8, 6, 8]; cutoff=0.0, α=3.0);
@time nnet.learn!()