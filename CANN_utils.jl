module CANN_utils

using CANN
using FANN

export encode, decode, load_pos_training, load_neg_training, preprocess
export conceptualize
export score
export rand_div, rand_n
export eval_nnets, write_predictions


## Takes a string of ATCGs and converts to binary as shown by encoder
##   Each nucleotide is represented by two binary digits
function encode(uncoded)
	encoder = {'A'=> [0., 0.], 'C'=> [0., 1.], 'G'=> [1., 0.], 'T'=> [1., 1.]}
	coded = []
	for ch in uncoded
		coded = vcat(coded, encoder[ch])
	end
	return coded'
end

## Takes a binary string and converts to ATCGs
function decode(coded)
	decoder = {[0., 0.] => 'A', [0., 1.] => 'C', [1., 0.] => 'G', [1., 1.] => 'T'}
	decoded = ""
	for ch in 1:2:length(coded)
		decoded = string(decoded, decoder[coded[ch:ch+1]])
	end
	return decoded
end

## Loads the positive data in 'file' (each line a sequence of ATCGs) and encodes each line
function load_pos_training(file)
	input = readdlm(file, '\t', ASCIIString)
	c_inp = zeros(Int64, length(input), length(input[1])*2)
	for i in 1:length(input)
		c_inp[i,:] = encode(input[i])
	end
	return float(c_inp)
end

## Loads a fasta file, takes up to 'num' sequences and randomly samples 'len' bp from each sequence
##  Also encodes each into a binary representation 
function load_neg_training(file; num = 1000, len = 17)
	input = readdlm(file, '\t', ASCIIString)
	seq = ""
	c_inp = zeros(Int64, num, len*2)
	c = 0
	for i in 2:length(input)
		line = input[i,1]
		if line[1] == '>'
			c += 1
			if c > num
				return float(c_inp)
			end
			# Get a random substring start
			start = rand(1:length(seq)-len)
			stop  = start+len
			uncoded = seq[start:stop-1]
			for ch in uncoded
				c_inp[c,:] = encode(uncoded)
			end
		else
			seq = string(seq,line)
		end
	end
end

## Given a nnet, will return the final layers activations for each x in 'xs'
function conceptualize(nnet::NNet, xs::Array{Float64,2})
	result = zeros(size(xs)[1], length(nnet.layers[2].a))
	for i in 1:size(xs)[1]
		nnet.predict(vec(xs[i,:]))
		result[i,:] = nnet.layers[2].a
	end
	return result
end

## Calculates TPR, TNR, FPR, FNR given a predictor function (nnet.predict())
##  Needs a dataset of true positives (TP) and true negatives (TN)
function score(predictor::Function, TP::Array{Float64,2}, TN::Array{Float64,2}, verbose::Bool)
	tp_n = size(TP)[1]
	tn_n = size(TN)[1]
	TPR = TNR = FPR = FNR = 0.0
	## True positives
	for i in 1:tp_n
		result = predictor(vec(TP[i,:]))
		result = round(result)
		if result[1] == 1.0
			TPR += 1.
		else
			FNR += 1.
		end
	end
	## True negatives
	for i in 1:tn_n
		result = predictor(vec(TN[i,:]))
		result = round(result)
		if result[1] == 0.0
			TNR += 1.
		else
			FPR += 1.
		end
	end
	TPR, FNR = [TPR, FNR]/tp_n
	TNR, FPR = [TNR, FPR]/tn_n
	#if verbose
		@printf("TPR: %8.3f    TNR: %8.3f    FPR: %8.3f    FNR: %8.3f\n", TPR, TNR, FPR, FNR)
	#end
	return [TPR, TNR, FPR, FNR]
end

## Will take a set of sequences in 'file', predict using a function (nnet.predict())
##  and write sequence, probability tab delimited 'outfile'
##  eg. ATATATCGTAGCAG	0.9242
function write_predictions(predictor::Function, file, outfile)
	testing = load_pos_training(file)  ## not pos training, but does similar parsing
	output  = open(outfile, "w")
	for i in 1:size(testing)[1]
		test = testing[i,:]
		pred = predictor(vec(test))[1]
		seq  = decode(test)
		write(output, "$(seq)\t$(pred)\n")
	end
	close(output)
end

## From the test set, will return new pos/neg training data above and below a threshold
function overfit(predictor::Function, file; thresh=.2)
	poss = float(zeros(1,34))
	negs = float(zeros(1,34))
	testing = load_pos_training(file)  ## not pos training, but does similar parsing
	for i in 1:size(testing)[1]
		test = testing[i,:]
		pred = predictor(vec(test))[1]
		if pred < thresh
			negs = vcat(negs, test)
		elseif pred > 1-thresh
			poss = vcat(poss, test)
		end
	end
	return (poss[2:end,:], negs[2:end,:])
end


# Randomize and then divide 'data' so that 'frac' of it is training, the rest testing
function rand_div(frac::Float64, data::Array{Float64,2})
	n = size(data)[1]
	i = [1:n]
	shuffle!(i)
	data = data[i,:]
	cutoff = int(round((frac)*n))
	data_training = data[1:cutoff, :]
	data_testing  = data[cutoff+1:end, :]
	return (data_training, data_testing)
end

# Randomize 'data' and then sample 'n' rows
function rand_n(n_::Int64, data::Array{Float64,2})
	n = size(data)[1]
	i = [1:n]
	shuffle!(i)
	data = data[i,:]
	data_sub = data[1:n_, :]
	return data_sub
end


function preprocess(poss::Array{Float64,2}, negs::Array{Float64,2}; hold_out=(1/3), nx = 1., use_all = false)
	if !use_all
		pos_training, pos_testing = rand_div(1-hold_out, poss);
		## Negative dataset, first subsetting to a ratio of the pos
		negs = rand_n(int(size(poss)[1]*nx), negs);
		neg_training, neg_testing = rand_div(1-hold_out, negs);
	else
		pos_training = poss
		neg_training = negs
		pos_testing  = []
		neg_testing  = []
	end
	training = vcat(pos_training, neg_training);
	classes  = vcat(ones(size(pos_training)[1]), zeros(size(neg_training)[1]));
	classes  = classes[1:length(classes),:];
	return (training, classes, pos_testing, neg_testing)
end

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
function eval_nnets(poss::Array{Float64,2}, negs::Array{Float64,2}; hold_out=(1/3), n_trials=20, sls=[34, 15, 34], alpha=0.9, iters=-1, cutoff=0.0001, nx=1, do_both=false, verbose=false)
	cann_scores = zeros(n_trials, 4);
	fann_scores = zeros(n_trials, 4);
	for t in 1:n_trials
		# preprocess the input data data
		training, classes, pos_testing, neg_testing = preprocess(poss, negs, hold_out=hold_out, nx=nx)
		
		## Train the classifier (9,000 iters appropriate?)
		classifier = NNet(training, classes, sls, alpha, iters, cutoff, verbose);
		classifier.learn!()
		cann_scores[t,:] = score(x->classifier.predict(x), pos_testing, neg_testing, verbose)
		# Sample
		# TPR:    1.000    TNR:    0.992    FPR:    0.000    FNR:    0.008

		if do_both
			## Convert half squared error to mean squared
			desired_error = (cutoff*2)
			# Modify the layer node numbers a bit (fann expects only one output node)
			sls[end] = 1
			# Make all the activating functions sigmoidal (for hidden layers)
			fcns = Array(Symbol, length(sls))
			for i in 1:length(sls)
				fcns[i] = :sigmoid_symmetric
			end
			dset = DataSet(training', classes')
			net = ANN([34, 5, 5, 1], fcns; b=0.1, errorfunc=:tanh, steepness=0.8)
			train!(net, dset, max_epochs=iters, epochs_between_reports=100, desired_error=desired_error)
			fann_scores[t,:] = score(x->predict(net, x), pos_testing, neg_testing, verbose)
			# Sample 
			# TPR:    1.000    TNR:    0.953    FPR:    0.000    FNR:    0.047
		end
	end
	if do_both
		return (cann_scores, fann_scores)
	else
		return cann_scores
	end
end


end