module rap1_utils


export encode, decode, load_pos_training, load_neg_training, write_predictions


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

## Will take a set of sequences in 'file', predict using a function (nnet.predict())
##  and write sequence, probability tab delimited 'outfile'
##  eg. ATATATCGTAGCAG	0.9242
function write_predictions(predictor::Function, file, outfile)
	testing = load_pos_training(file)
	output  = open(outfile, "w")
	for i in 1:size(testing)[1]
		test = testing[i,:]
		pred = predictor(vec(test))[1]
		seq  = decode(test)
		write(output, "$(seq)\t$(pred)\n")
	end
	close(output)
end

end # of module