#!/bin/env julia
using CANN          # My neural network implementation
using CANN_utils    # Functions used to process rap1 data
using rap1_utils
using FANN          # The Fast Neural Network C code wrapper for julia


#########################
#--- Load the data
#########################
poss = load_pos_training("rap1-lieb-positives.txt");
negs = load_neg_training("yeast-upstream-1k-negative.fa", num=3000);
prediction_file = "rap1-lieb-test.txt"

### Neural Network evaluation function
###  poss      The positive training set
###  negs      The negative training set
###  hold_out  The % of each dataset to hold for testing (the rest is used to train)
###  n_trials  The number of times cross-validation is performed
###  sls       The number of nodes in each layer of the neural network
###  alpha     The learning rate of the nnet
###  iters     The maximum number of learning iterations
###  cutoff    The level of learning (half squared error) required before stopping
###
###  Outputs a list of TPR, TNR, FPR, FNR for both my nnet implementation and FANN (C library) for each trial 
#(cann_scores, fann_scores) = eval_nnets(poss, negs; hold_out=hold_out, n_trials=20, sls=[34, 15, 34], alpha=0.9, iters=20000, cutoff=0.0001)
n_trials = 20
## Determine how changing alpha (learning rate) affects performance
for alpha in [0.03, 0.3, 3.0, 30.]
	cann_scores = eval_nnets(poss, negs; hold_out=.5, n_trials=n_trials, sls=[34, 15, 34], alpha=alpha, iters=50000, cutoff=0.1)
	cann_means = mean(cann_scores,1)
	cann_stds  = std(cann_scores,1)
	println("$(round(cann_means,3))\n$(round(cann_stds,3))")
end

## Determine how changing the hold_out (for testing) affects performance
for hold_out in [1/10, 1/4, 1/3, 1/2, 2/3, 3/4, 9/10]
	cann_scores = eval_nnets(poss, negs; hold_out=hold_out, n_trials=n_trials, sls=[34, 15, 34], alpha=0.9, iters=50000, cutoff=0.001)
	cann_means = mean(cann_scores,1)
	cann_stds  = std(cann_scores,1)
	println("$(round(cann_means,3))\n$(round(cann_stds,3))")
end

## Determine how changing the nnet arch. affects performance
for l in [30, 25, 20, 15,10, 5]
	cann_scores = eval_nnets(poss, negs; hold_out=.5, n_trials=n_trials, sls=[34, l, 34], alpha=0.9, iters=50000, cutoff=0.001)
	cann_means = mean(cann_scores,1)
	cann_stds  = std(cann_scores,1)
	println("$(round(cann_means,3))\n$(round(cann_stds,3))")
end

## Determine how changing ratio between pos and neg training examples affects performance
for l in [.25, .5, 1, 5, 10, 20]
	cann_scores = eval_nnets(poss, negs; hold_out=.5, n_trials=n_trials, sls=[34, 15, 34], alpha=0.9, iters=50000, cutoff=0.001, nx=l)
	cann_means = mean(cann_scores,1)
	cann_stds  = std(cann_scores,1)
	println("$(round(cann_means,3))\n$(round(cann_stds,3))")
end


### Algorithm Comparison
(cann_scores,fann_scores) = eval_nnets(poss, negs; hold_out=(1/3), n_trials=50, sls=[34, 20, 34], alpha=0.9, iters=1000000, cutoff=0.0001, verbose=true, do_both=true)


### Best parameters and best algorithm 
training, classes, pos_testing, neg_testing = preprocess(poss, negs, hold_out=0.5, nx=5);
## Train the classifier (9,000 iters appropriate?)
best_classifier = NNet(training, classes, [34, 20, 1], 4., 50000, 0.0000001, true);
best_classifier.learn!()
score(x->best_classifier.predict(x), pos_testing, neg_testing, true)
#score(x->best_classifier.predict(x), poss, negs, true)
### Write the output
write_predictions(x->best_classifier.predict(x), prediction_file, "Cario_Clint_BMI203_final_predictions.txt")



################################
#--- Alternative Trainer
################################
## FANN way, but no way to access final hidden layer
#dset = DataSet(negs', negs')
#net = ANN([34, 20, 34], [:sigmoid_symmetric, :sigmoid_symmetric]; b=0.1, errorfunc=:tanh, steepness=19.5)
#@time train!(net, dset, max_epochs=1000000, epochs_between_reports=1, desired_error=0.001)
# Rotate through
neg_concept = NNet(negs[1:1000,:], negs[1:1000,:], [34, 25, 34], 7.5, int(5e4), 0., true);
#for _ in 1:3
#	for i in 1:100:1000
#		println("Burning set [$(i):$(i+100)]...")
#		neg_concept.learn!(features = negs[i:i+100,:], targets  = negs[i:i+100,:])     # Negative concepts are harder to learn, probably due to more random nature
#	end
#end
#neg_concept.learn!(features=negs[1:1000,:], targets=negs[1:1000,:], α=7.0, iters=int(5e5));
neg_concept.learn!()

pos_concept = NNet(poss, poss, [34, 34, 34], 4.0, int(1e6), 0., true);
pos_concept.learn!()     # Cost of 0.01 at this point usually


# Run the positive and negative sets through both conceptulizers, combine for training input
neg_conc = conceptualize(pos_concept, negs);
neg_conc = hcat(neg_conc, conceptualize(neg_concept, negs));
pos_conc = conceptualize(pos_concept, poss);
pos_conc = hcat(pos_conc, conceptualize(neg_concept, poss));


# Process the training/testing data
training, classes, pos_testing, neg_testing = preprocess(pos_conc, neg_conc, hold_out=(1/2));
## Train the classifier and then test it 
alt = NNet(training, classes, [40, 25, 1], 4.0, -1, 0.0000001, true);
alt.learn!();

## Need to redo to get testing data that is from original dataset (non conceptualized)
training, classes, pos_testing, neg_testing = preprocess(poss, negs, hold_out=(1/2));
pos_test = conceptualize(pos_concept, pos_testing);
pos_test = hcat(pos_test, conceptualize(neg_concept, pos_testing));
neg_test = conceptualize(pos_concept, neg_testing);
neg_test = hcat(neg_test, conceptualize(neg_concept, neg_testing));
#score(x->alt.predict(x), conceptualize(pos_concept, pos_testing), conceptualize(neg_concept, neg_testing), true)
score(x->alt.predict(x), pos_test, neg_test, true)


#### NEED TO PASS THROUGH CONCEPTULIZERS FIRST
## Write the output
#write_predictions(x->alt.predict(x), prediction_file, "Cario_Clint_BMI203_alt.txt")

