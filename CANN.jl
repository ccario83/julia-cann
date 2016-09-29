module CANN

using Distributions
#using Debug

export sigmoid
export sigmoid′
export half_squared_cost
export mean_squared_error
export Layer
export NNet
export save_nnet
export load_nnet

export layer_info
export ppvec
export ppmat

# Symbols used
# ∇ ′ α Θ Δ δ λ Σ

# New tanh and sech function overrides
import Base.tanh, Base.sech
tanh(z::Vector{Float64}, r::Vector{Float64}) = (r[:] = tanh(z); nothing)
sech(z::Vector{Float64}, r::Vector{Float64}) = (r[:] = sech(z); nothing)
# Note: z is a vector of Θ'*a, where a is the activation states from previous layer and Θ is the corresponding weights
sigmoid(z::Vector{Float64}, r::Vector{Float64})  = (r[:] = 1./(1+exp(-z)); nothing)
sigmoid′(a::Vector{Float64}, r::Vector{Float64}) = (r[:] = a.*(1-a); nothing)
#sigmoid′(z::Vector{Float64}, result::Vector{Float64}) = exp(z)./(exp(z)+1).^2
half_squared_cost(y_hat::Vector{Float64}, y::Vector{Float64}) = 0.5*sum((y_hat-y).^2)
half_squared_cost_rounded(y_hat::Vector{Float64}, y::Vector{Float64}) = 0.5*sum((round(y_hat)-y).^2)
mean_squared_error(y_hat::Vector{Float64}, y::Vector{Float64}) = mean((y_hat-y).^2)

type Layer
	## State Variables
	x::Vector{Float64}		# The inputs from previous layer
	Θ::Array{Float64, 2}	# The weights
	b::Vector{Float64}		# The biases
	z::Vector{Float64}		# The pre-activation state of each node
	a::Vector{Float64}		# The activation state of each node
	
	## Backpropagation variables
	ΔΘ::Array{Float64, 2}	# Accumulation variable for the weight gradient
	Δb::Vector{Float64}		# Accumulation variable for biases gradient

	## Firing functions
	g::Function
	g′::Function

	function Layer(slp::Int64, sl::Int64, g::Function, g′::Function)
		# Initialize state variables 
		x = float(zeros(slp))				# Input states
		a = float(zeros(sl))				# Output states
		Θ = rand(Normal(0,.0001),(sl,slp))	# Near zero weights
		z = float(zeros(sl))				# Pre-activation values
		b = float(ones(sl))					# The biases for each node in this layer

		# Backpropagation variable initialization
		ΔΘ = float(zeros(sl,slp))
		Δb = float(zeros(sl))

		new(x, Θ, b, z, a, ΔΘ, Δb, g, g′)
	end
end
Layer(sl     ; g=sigmoid, g′=sigmoid′) = Layer(sl,  sl, g, g′)
Layer(slp, sl; g=sigmoid, g′=sigmoid′) = Layer(slp, sl, g, g′)


type NNet
	## Input/Output 
	# The feature and target data is similar, and structured like so (floats):
	#[	
	#	x1(1)	x2(1)	...	xm(1)
	#	x1(2)	x2(2)	...	xm(2)
	#					...
	#	x1(n)	x2(n)	...	xm(n)
	#]
	features::Array{Float64,2}
	targets::Array{Float64,2}
	
	## Layer info
	L::Int64					# The number of layers (including input and output layers)
	sls::Vector{Int64}			# The number of nodes in each layer
	layers::Vector{Layer}		# A vector of layers consisting of the nodes (vectorized)
	
	## Functions
	g::Function					# The activation function
	g′::Function				# The activation function derivative
	h::Function 				# The final output function 
	J::Function 				# The partial cost function (will be summed over x(i))
	
	# Learning parameters
	α::Float64					# Learning rate
	λ::Float64					# Weight decay
	
	# Internal functions
	learn!::Function
	forward!::Function
	backpropagate!::Function
	regularization::Function
	predict::Function

	# learning params
	iters::Int64
	cutoff::Float64
	verbose::Bool

	# The initialization function 
	function NNet(features::Array{Float64,2}, targets::Array{Float64,2}, sls::Vector{Int64}, g::Function, g′::Function, h::Function, J::Function, α::Float64, λ::Float64, iters::Int64, cutoff::Float64, verbose::Bool)
		## Define input and output sizes
		m, n = size(features)
		o    = size(targets)[2]
		L    = int(length(sls))

		if L < 3
			print("You must specify more than 3 layers (including input and output)")
			return
		end
		if n != sls[1]
			print("The first layer should have as many nodes as there are inputs or features in the training data")
			return
		end
		if o != sls[end] 
			print("The final layer should have as many nodes as there are outputs or classes in the training data")
			return
		end

		## Initialize the layers
		layers    = Array(Layer, L)
		layers[1] = Layer(1, n, g, g′)		# Layer 1 is just inputs
		slp       = n 						# Adapts layer 1 to first hidden layer
		
		for l in 2:L-1
			sl = sls[l]
			layers[l] = Layer(slp, sl, g, g′)
			slp = sl
		end
		layers[L] = Layer(slp, o, g, g′)


		function learn!(;features::Array{Float64,2}=features, targets::Array{Float64,2}=targets, α::Float64=α, λ::Float64=λ, iters::Int64=iters, cutoff::Float64=cutoff, verbose::Bool=verbose)
			if verbose
				print("Learning...")
			end
			# The number of learning iterations (count)
			c = 0
			t_cost  = 1
			t_error = 1
			error  = zeros(o)
			y_hats = zeros(o)

			while (t_cost > cutoff) && (t_error != 0.0) && (c < iters || iters == -1)
				c += 1
				t_cost  = 0
				t_error = zeros(n)

				## For each training example
				for i in 1:m
					xs     = vec(features[i,:])
					ys     = vec(targets[i,:])
					forward!(xs, y_hats)			# Using prealloc'd y_hates
					error  = -(ys - y_hats)
					backpropagate!(error)
					#println(y_hats)
					#println(ys)
					#@bp

					# Σ[1,m] part of the cost function, and the cumulative error
					t_cost  += J(y_hats, ys)
					#t_error += abs(error)
				end
				#if c >= 7000
				#	layer_info(layers)
				#	@bp
				#end

				## Update weights and continue
				for li in 2:L
					l = layers[li]
					l.Θ += -α * ((1/m * l.ΔΘ) + (λ*l.Θ))
					l.b += -α *  (1/m * l.Δb)
					# Reset accumulators 
					l.ΔΘ -= l.ΔΘ
					l.Δb -= l.Δb
				end

				## Adjust the partial cost and error to full versions 
				t_cost  *= 1/m
				t_cost  += (λ != 0.0? regularization() : 0)  # Faster to just set to 0
				#t_error *= 1/m
				if c%50 == 1 && verbose
					@printf("\n%10d:   Cost: %e  ", c, t_cost)
					#layer_info(layers)
				end
			end
		end

		function forward!(xs::Vector{Float64}, y_hats::Vector{Float64})
			## The first layer
			ap = layers[1].a = xs

			## Each hidden layer
			for li in 2:L-1
				l = layers[li] 			# Get the current layer
				l.x = ap 				# Set the nodes input
				l.z = l.Θ * l.x + l.b 	# Calculate the pre-activation value
				
				g(l.z, l.a) 			# Activate, store as l.a
				
				ap = l.a 				# This activation is the next layers input
			end

			## The output layer
			l = layers[L]
			l.x = ap
			l.z = l.Θ * l.x + l.b
			h(l.z, l.a) 				# Use the h function here instead

			y_hats[:] = l.a 			# Store the results in y_hats (preallocated)
			nothing
		end

		#function backpropagate!(error)
		#	δ = error
		#	for l in L:-1:2
		#		layers[l].ΔΘ += δ * layers[l].x'
		#		layers[l].Δb += δ
		#		δn = layers[l].Θ' * δ
		#		δ = δn .* layers[l-1].g′(layers[l-1].z)
		#	end
		#	layers[1].ΔΘ += δ * layers[1].x'
		#	layers[1].Δb += δ
		#end

		function backpropagate!(error)
			δn = error
			for li in L:-1:1
				l = layers[li]
				#δn = δn .* l.g′(l.z)
				r = zeros(length(l.a))
				l.g′(l.a, r)
				δn = δn .* r

				l.ΔΘ += δn * l.x'
				l.Δb += δn

				δn = l.Θ' * δn
			end
		end

		function regularization()
			reg = 0.
			# In each layer but the last, sum square the weights (including bias?)
			for l in 2:L
				reg += sum(layers[l].Θ.^2)+sum(layers[l].b.^2)
			end
			# Scale reg by lambda over two
			reg *= λ/2
			return reg
		end

		function predict(xs::Vector{Float64})
			y_hats = zeros(o)
			forward!(xs, y_hats)
			return y_hats
		end
		new(features, targets, L, sls, layers, g, g′, h, J, α, λ, learn!, forward!, backpropagate!, regularization, predict, iters, cutoff, verbose)
	end
end
## Various ways to call the NNet
NNet(features, targets; sls=[8,3,8],                      g=sigmoid, g′=sigmoid′, h=sigmoid, J=half_squared_cost, α=0.9, λ=0.0, iters=-1, cutoff=0.0001, verbose=true) = NNet(features, targets, sls, g, g′, h, J, α, λ, iters, cutoff, verbose)
NNet(features, targets, sls;                              g=sigmoid, g′=sigmoid′, h=sigmoid, J=half_squared_cost, α=0.9, λ=0.0, iters=-1, cutoff=0.0001, verbose=true) = NNet(features, targets, sls, g, g′, h, J, α, λ, iters, cutoff, verbose)
NNet(features, targets, sls, α, iters::Int64;             g=sigmoid, g′=sigmoid′, h=sigmoid, J=half_squared_cost, λ=0.0, cutoff=0.0, verbose=true)                     = NNet(features, targets, sls, g, g′, h, J, α, λ, iters, cutoff, verbose)
NNet(features, targets, sls, iters::Int64;                g=sigmoid, g′=sigmoid′, h=sigmoid, J=half_squared_cost, α=0.9, λ=0.0, cutoff=0.0, verbose=true)              = NNet(features, targets, sls, g, g′, h, J, α, λ, iters, cutoff, verbose)
NNet(features, targets, sls, α, cutoff::Float64;          g=sigmoid, g′=sigmoid′, h=sigmoid, J=half_squared_cost, λ=0.0, iters=-1, verbose=true)                       = NNet(features, targets, sls, g, g′, h, J, α, λ, iters, cutoff, verbose)
NNet(features, targets, sls, cutoff::Float64;             g=sigmoid, g′=sigmoid′, h=sigmoid, J=half_squared_cost, α=0.9, λ=0.0, iters=-1, verbose=true)                = NNet(features, targets, sls, g, g′, h, J, α, λ, iters, cutoff, verbose)
NNet(features, targets, sls, α, iters, cutoff, verbose;   g=sigmoid, g′=sigmoid′, h=sigmoid, J=half_squared_cost, λ=0.0)                                               = NNet(features, targets, sls, g, g′, h, J, α, λ, iters, cutoff, verbose)


######### DEBUG ##############
function layer_info(nnet::NNet)
	layers = nnet.layers
	@printf("%103s\n",repeat("_",103))
	println("Activations:")
	@printf("%103s\n",repeat("_",103))
	max_nodes = 0
	for layer in layers
		if size(layer.Θ)[1] > max_nodes
			max_nodes = size(layer.Θ)[1]
		end
	end

	for line in 1:max_nodes
		for layer in layers
			if line > size(layer.Θ)[1]
				@printf("%10s","")
			else
				@printf("%10.5f",layer.a[line])
			end
		end
		println()
	end

	@printf("%103s\n",repeat("_",103))
	println("Θ & b:")
	@printf("%103s\n",repeat("_",103))
	for t in 2:size(layers)[1]
		Θ = layers[t].Θ
		b = layers[t].b
		for i in 1:size(Θ)[1]
			for j in 1:size(Θ)[2]
				@printf("%12.4f", Θ[i,j])
			end
			println()
		end

		@printf("%96s\n",repeat("-",90))
		for j in 1:length(b)
			@printf("%12.4f", b[j])
		end
		if t != (size(layers)[1])
			@printf("\n%96s\n",repeat("=",90))
		else
			println()
		end
	end

	@printf("%103s\n",repeat("_",103))
	println("ΔΘ Δb:")
	@printf("%103s\n",repeat("_",103))
	for t in 2:size(layers)[1]
		Θ = layers[t].ΔΘ
		b = layers[t].Δb
		for i in 1:size(Θ)[1]
			for j in 1:size(Θ)[2]
				@printf("%12.4f", Θ[i,j])
			end
			println()
		end
		
		@printf("%96s\n",repeat("-",90))
		for j in 1:length(b)
			@printf("%12.4f", b[j])
		end
		if t != (size(layers)[1])
			@printf("\n%96s\n",repeat("=",90))
		else
			println()
		end
	end
	#println("=======================")
end

function ppmat(mat)
	a, b = size(mat)
	for i in 1:a
		for j in 1:b
			@printf("%10.4f", mat[i,j])
		end
		println()
	end
end

function ppvec(v)
	b = length(v)
	print(" [")
	for j in 1:b
		@printf("%3.5f ", v[j])
	end
	print("]\n")
end
###############################

end

