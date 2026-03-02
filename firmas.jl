
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # Primero se comprueba que todos los elementos del vector esten en el vector de clases (linea adaptada del final del ejercicio 4)
	@assert(all([in(value, classes) for value in feature]))
	numClasses = length(classes)
	if (numClasses <= 2)
		# Si solo hay dos clases, se devuelve una matriz con una columna
		oneHot = reshape(feature .== classes[1], :, 1)
	else
		# Si hay mas de dos clases se devuelve una matriz con una columna por clase
		oneHot = convert(BitArray{2}, hcat([instance .== classes for instance in feature]...)')
	end
	return oneHot
end;

oneHotEncoding(feature::AbstractArray{<:Any, 1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);

calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real, 2}) = (minimum(dataset, dims = 1), maximum(dataset, dims = 1));

calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2}) = (mean(dataset, dims = 1), std(dataset, dims = 1));

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1]
    maxValues = normalizationParameters[2]
    dataset .-= minValues
    dataset ./= (maxValues .- minValues)
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    dataset[:, vec(minValues .== maxValues)] .= 0
    return dataset
end;
normalizeMinMax!(dataset::AbstractArray{<:Real, 2}) = normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset));
normalizeMinMax(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}}) = normalizeMinMax!(copy(dataset), normalizationParameters)
normalizeMinMax(dataset::AbstractArray{<:Real, 2}) = normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));

# 4 versiones similares de la funcion para normalizar de media 0:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeZeroMean!(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
	avgValues = normalizationParameters[1]
	stdValues = normalizationParameters[2]
	dataset .-= avgValues
	dataset ./= stdValues
	# Si hay algun atributo en el que todos los valores son iguales, se pone a 0
	dataset[:, vec(stdValues .== 0)] .= 0
	return dataset
end;
normalizeZeroMean!(dataset::AbstractArray{<:Real, 2}) = normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset));
normalizeZeroMean(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}}) = normalizeZeroMean!(copy(dataset), normalizationParameters)
normalizeZeroMean(dataset::AbstractArray{<:Real, 2}) = normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));


# -------------------------------------------------------
# Funcion que permite transformar una matriz de valores reales con las salidas del clasificador o clasificadores en una matriz de valores booleanos con la clase en la que sera clasificada

classifyOutputs(outputs::AbstractArray{<:Real, 1}; threshold::Real = 0.5) = outputs .>= threshold;

function classifyOutputs(outputs::AbstractArray{<:Real, 2}; threshold::Real = 0.5)
	numOutputs = size(outputs, 2)
	@assert(numOutputs != 2)
	if numOutputs == 1
		return reshape(classifyOutputs(outputs[:]; threshold = threshold), :, 1)
	else
		# Miramos donde esta el valor mayor de cada instancia con la funcion findmax
		(_, indicesMaxEachInstance) = findmax(outputs, dims = 2)
		# Creamos la matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
		outputs = falses(size(outputs))
		outputs[indicesMaxEachInstance] .= true
		# Comprobamos que efectivamente cada patron solo este clasificado en una clase
		@assert(all(sum(outputs, dims = 2) .== 1))
		return outputs
	end
end;


# -------------------------------------------------------
# Funciones para calcular la precision

accuracy(outputs::AbstractArray{Bool, 1}, targets::AbstractArray{Bool, 1}) = mean(outputs .== targets);
function accuracy(outputs::AbstractArray{Bool, 2}, targets::AbstractArray{Bool, 2})
	@assert(all(size(outputs) .== size(targets)))
	if (size(targets, 2) == 1)
		return accuracy(outputs[:, 1], targets[:, 1])
	else
		return mean(all(targets .== outputs, dims = 2))
	end
end;

accuracy(outputs::AbstractArray{<:Real, 1}, targets::AbstractArray{Bool, 1}; threshold::Real = 0.5) = accuracy(outputs .>= threshold, targets);
function accuracy(outputs::AbstractArray{<:Real, 2}, targets::AbstractArray{Bool, 2}; threshold::Real = 0.5)
	@assert(all(size(outputs) .== size(targets)))
	if (size(targets, 2) == 1)
		return accuracy(outputs[:, 1], targets[:, 1]; threshold = threshold)
	else
		return accuracy(classifyOutputs(outputs), targets)
	end
end;


# -------------------------------------------------------
# Funciones para crear y entrenar una RNA

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int, 1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function, 1} = fill(σ, length(topology)))
	ann = Chain()
	numInputsLayer = numInputs
	for numHiddenLayer in 1:length(topology)
		numNeurons = topology[numHiddenLayer]
		ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]))
		numInputsLayer = numNeurons
	end
	if (numOutputs == 1)
		ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
	else
		ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
		ann = Chain(ann..., softmax)
	end
	return ann
end;

function trainClassANN(
	topology::AbstractArray{<:Int, 1},
	dataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}};
	transferFunctions::AbstractArray{<:Function, 1} = fill(σ, length(topology)),
	maxEpochs::Int = 1000,
	minLoss::Real = 0.0,
	learningRate::Real = 0.01,
)

	(inputs, targets) = dataset

	# Se supone que tenemos cada patron en cada fila
	# Comprobamos que el numero de filas (numero de patrones) coincide
	@assert(size(inputs, 1) == size(targets, 1))

	# Pasamos los datos a Float32
	inputs = Float32.(inputs)

	# Creamos la RNA
	ann = buildClassANN(size(inputs, 2), topology, size(targets, 2))

	# Definimos la funcion de loss
	loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

	# Creamos los vectores con los valores de loss y de precision en cada ciclo
	trainingLosses = Float32[]

	# Empezamos en el ciclo 0
	numEpoch = 0
	# Calculamos el loss para el ciclo 0 (sin entrenar nada)
	trainingLoss = loss(ann, inputs', targets')
	#  almacenamos el valor de loss y precision en este ciclo
	push!(trainingLosses, trainingLoss)
	#  y lo mostramos por pantalla
	println("Epoch ", numEpoch, ": loss: ", trainingLoss)

	opt_state = Flux.setup(Adam(learningRate), ann)

	# Entrenamos hasta que se cumpla una condicion de parada
	while (numEpoch < maxEpochs) && (trainingLoss > minLoss)

		# Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
		Flux.train!(loss, ann, [(inputs', targets')], opt_state)

		# Aumentamos el numero de ciclo en 1
		numEpoch += 1
		# Calculamos las metricas en este ciclo
		trainingLoss = loss(ann, inputs', targets')
		#  almacenamos el valor de loss
		push!(trainingLosses, trainingLoss)
		#  lo mostramos por pantalla
		println("Epoch ", numEpoch, ": loss: ", trainingLoss)

	end

	# Devolvemos la RNA entrenada y el vector con los valores de loss
	return (ann, trainingLosses)
end;


trainClassANN(
	topology::AbstractArray{<:Int, 1},
	(inputs, targets)::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 1}};
	transferFunctions::AbstractArray{<:Function, 1} = fill(σ, length(topology)),
	maxEpochs::Int = 1000,
	minLoss::Real = 0.0,
	learningRate::Real = 0.01,
) = trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate)


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
	@assert ((P >= 0.0) & (P <= 1.0))
	indices = randperm(N)
	numTrainingInstances = Int(round(N * (1 - P)))
	return (indices[1:numTrainingInstances], indices[numTrainingInstances+1:end])
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
	@assert ((Pval >= 0.0) & (Pval <= 1.0))
	@assert ((Ptest >= 0.0) & (Ptest <= 1.0))
	@assert ((Pval + Ptest) <= 1.0)
	# Primero separamos en entrenamiento+validation y test
	(trainingValidationIndices, testIndices) = holdOut(N, Ptest)
	# Después separamos el conjunto de entrenamiento+validation
	(trainingIndices, validationIndices) = holdOut(length(trainingValidationIndices), Pval * N / length(trainingValidationIndices))
	return (trainingValidationIndices[trainingIndices], trainingValidationIndices[validationIndices], testIndices)
end;



# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test. Estos dos ultimos son opcionales
# Es la funcion anterior, modificada para calcular errores en los conjuntos de validacion y test y realizar parada temprana si es necesario
function trainClassANN(topology::AbstractArray{<:Int, 1},
	trainingDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}};
	validationDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}} = (Array{eltype(trainingDataset[1]), 2}(undef, 0, size(trainingDataset[1], 2)), falses(0, size(trainingDataset[2], 2))),
	testDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}}       = (Array{eltype(trainingDataset[1]), 2}(undef, 0, size(trainingDataset[1], 2)), falses(0, size(trainingDataset[2], 2))),
	transferFunctions::AbstractArray{<:Function, 1}                            = fill(σ, length(topology)),
	maxEpochs::Int                                                             = 1000, minLoss::Real                                                              = 0.0, learningRate::Real                                                         = 0.01,
	maxEpochsVal::Int                                                          = 20)

	(trainingInputs, trainingTargets)     = trainingDataset
	(validationInputs, validationTargets) = validationDataset
	(testInputs, testTargets)             = testDataset

	# Se supone que tenemos cada patron en cada fila
	# Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en validacion como test
	@assert(size(trainingInputs, 1) == size(trainingTargets, 1))
	@assert(size(testInputs, 1) == size(testTargets, 1))
	@assert(size(validationInputs, 1) == size(validationTargets, 1))
	# Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y validación, si este no está vacío
	!isempty(validationInputs) && @assert(size(trainingInputs, 2) == size(validationInputs, 2))
	!isempty(validationTargets) && @assert(size(trainingTargets, 2) == size(validationTargets, 2))
	# Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y test, si este no está vacío
	!isempty(testInputs) && @assert(size(trainingInputs, 2) == size(testInputs, 2))
	!isempty(testTargets) && @assert(size(trainingTargets, 2) == size(testTargets, 2))

	# Pasamos los datos a Float32
	trainingInputs   = Float32.(trainingInputs)
	validationInputs = Float32.(validationInputs)
	testInputs       = Float32.(testInputs)

	# Creamos la RNA
	ann = buildClassANN(size(trainingInputs, 2), topology, size(trainingTargets, 2); transferFunctions = transferFunctions)

	# Definimos la funcion de loss
	loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

	# Creamos los vectores con los valores de loss y de precision en cada ciclo
	trainingLosses   = Float32[]
	validationLosses = Float32[]
	testLosses       = Float32[]

	# Empezamos en el ciclo 0
	numEpoch = 0

	# Una funcion util para calcular los resultados y mostrarlos por pantalla si procede
	function calculateLossValues()
		# Calculamos el loss en entrenamiento, validacion y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
		trainingLoss = loss(ann, trainingInputs', trainingTargets')
		validationLoss = NaN
		testLoss = NaN
		push!(trainingLosses, trainingLoss)
		!isempty(validationInputs) && (validationLoss = loss(ann, validationInputs', validationTargets'); push!(validationLosses, validationLoss))
		!isempty(testInputs) && (testLoss = loss(ann, testInputs', testTargets'); push!(testLosses, testLoss))
		return (trainingLoss, validationLoss, testLoss)
	end

	# Calculamos los valores de loss para el ciclo 0 (sin entrenar nada)
	(trainingLoss, validationLoss, _) = calculateLossValues()

	if isempty(validationInputs)
		maxEpochsVal = Inf
	end

	# Numero de ciclos sin mejorar el error de validacion y el mejor error de validation encontrado hasta el momento
	numEpochsValidation = 0
	bestValidationLoss = validationLoss
	# Cual es la mejor ann que se ha conseguido
	bestANN = deepcopy(ann)

	opt_state = Flux.setup(Adam(learningRate), ann)

	# Entrenamos hasta que se cumpla una condicion de parada
	while (numEpoch < maxEpochs) && (trainingLoss > minLoss) && (numEpochsValidation < maxEpochsVal)

		# Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
		Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt_state)

		# Aumentamos el numero de ciclo en 1
		numEpoch += 1

		# Calculamos los valores de loss para este ciclo
		(trainingLoss, validationLoss, _) = calculateLossValues()

		# Aplicamos la parada temprana si hay conjunto de validacion
		if !isempty(validationInputs)
			if validationLoss < bestValidationLoss
				bestValidationLoss = validationLoss
				numEpochsValidation = 0
				bestANN = deepcopy(ann)
			else
				numEpochsValidation += 1
			end
		end

	end

	# Si no hubo conjunto de validacion, la mejor RNA será siempre la del último ciclo
	if isempty(validationInputs)
		bestANN = ann
	end

	return (bestANN, trainingLosses, validationLosses, testLosses)
end;



function trainClassANN(topology::AbstractArray{<:Int, 1},
	trainingDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 1}};
	validationDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 1}} = (Array{eltype(trainingDataset[1]), 2}(undef, 0, size(trainingDataset[1], 2)), falses(0)),
	testDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 1}}       = (Array{eltype(trainingDataset[1]), 2}(undef, 0, size(trainingDataset[1], 2)), falses(0)),
	transferFunctions::AbstractArray{<:Function, 1}                            = fill(σ, length(topology)),
	maxEpochs::Int                                                             = 1000, minLoss::Real                                                              = 0.0, learningRate::Real                                                         = 0.01,
	maxEpochsVal::Int                                                          = 20)

	(trainingInputs, trainingTargets)     = trainingDataset
	(validationInputs, validationTargets) = validationDataset
	(testInputs, testTargets)             = testDataset

	return trainClassANN(
		topology,
		(trainingInputs, reshape(trainingTargets, length(trainingTargets), 1));
		validationDataset = (validationInputs, reshape(validationTargets, length(validationTargets), 1)),
		testDataset = (testInputs, reshape(testTargets, length(testTargets), 1)),
		transferFunctions = transferFunctions,
		maxEpochs = maxEpochs,
		minLoss = minLoss,
		learningRate = learningRate,
		maxEpochsVal = maxEpochsVal,
	)
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum((outputs .== true) .& (targets .== true)) # Verdaderos Positivos
    VN = sum((outputs .== false) .& (targets .== false)) # Verdaderos Negativos
    FP = sum((outputs .== true) .& (targets .== false)) # Falsos Positivos
    FN = sum((outputs .== false) .& (targets .== true)) # Falsos Negativos

    accuracy = (VP + VN) / (VP + VN + FP + FN)
    error_rate = (FP + FN) / (VP + VN + FP + FN) # 1. - accuracy;?
    sensitivity = VP == 0 && FN == 0 ? 1.0 : ( VP / (VP + FN)) # Sensibilidad si VP y FN son 0
    specificity = VN == 0 && FP == 0 ? 1.0 : (VN / (VN + FP)) # Especificidad si VN y FP son 0
    precision = VP == 0 && FP == 0 ? 1.0 : (VP / (VP + FP)) # Precisión si VP y FP son 0
    NPV = VN == 0 && FN == 0 ? 1.0 : (VN / (VN + FN)) # Valor predictivo negativo si VN y FN son 0
    f1_score = precision == 0 && sensitivity == 0 ? 0 : 2 * (precision * sensitivity) / (precision + sensitivity)
    confusion_matrix = [VN FP; FN VP]

    return accuracy, error_rate, sensitivity, specificity, precision, NPV, f1_score, confusion_matrix
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .>= threshold
    confusionMatrix(outputs, targets)
end;

function confusionMatrix(outputs::AbstractArray{Bool, 2}, targets::AbstractArray{Bool, 2}; weighted::Bool = true)
	@assert(size(outputs) == size(targets))
	(numInstances, numClasses) = size(targets)
    
	# Nos aseguramos de que no hay dos columnas
	@assert(numClasses != 2)
	if (numClasses == 1)
		return confusionMatrix(outputs[:, 1], targets[:, 1])
	end

	# Nos aseguramos de que en cada fila haya uno y solo un valor a true
	@assert(all(sum(outputs, dims = 2) .== 1))
    
	# Metricas
	resultados = confusionMatrix.(eachcol(outputs), eachcol(targets))
	
	recall      = getindex.(resultados, 3)
	specificity = getindex.(resultados, 4)
	precision   = getindex.(resultados, 5)
	NPV         = getindex.(resultados, 6)
	F1          = getindex.(resultados, 7)

	confMatrix = targets' * outputs # 

	# Aplicamos las formas de combinar las metricas macro o weighted
	if weighted
		# Calculamos los valores de ponderacion para hacer el promedio
		numInstancesFromEachClass = vec(sum(targets, dims = 1))
		@assert(numInstances == sum(numInstancesFromEachClass))
		weights     = numInstancesFromEachClass ./ sum(numInstancesFromEachClass)
        
		recall      = sum(weights .* recall)
		specificity = sum(weights .* specificity)
		precision   = sum(weights .* precision)
		NPV         = sum(weights .* NPV)
		F1          = sum(weights .* F1)
	else
		recall      = mean(recall)
		specificity = mean(specificity)
		precision   = mean(precision)
		NPV         = mean(NPV)
		F1          = mean(F1)
	end
    
	# Precision y tasa de error las calculamos con las funciones definidas previamente
	acc = accuracy(outputs, targets) # Asumo que tienes esta función en tu entorno
	errorRate = 1 - acc

	return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end
confusionMatrix(outputs::AbstractArray{<:Real, 2}, targets::AbstractArray{Bool, 2}; threshold::Real = 0.5, weighted::Bool = true) = confusionMatrix(classifyOutputs(outputs; threshold = threshold), targets; weighted = weighted)

function confusionMatrix(outputs::AbstractArray{<:Any, 1}, targets::AbstractArray{<:Any, 1}, classes::AbstractArray{<:Any, 1}; weighted::Bool = true)
	# Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
	@assert(all([in(label, classes) for label in vcat(targets, outputs)]))
	# Es importante pasar el mismo vector de clases como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
	return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted = weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any, 1}, targets::AbstractArray{<:Any, 1}; weighted::Bool = true)
	classes = unique(vcat(targets, outputs))
	return confusionMatrix(outputs, targets, classes; weighted = weighted)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Llamamos a la funcion que ya desarrollaste para obtener todas las metricas
    accuracy, error_rate, sensitivity, specificity, precision_val, NPV, f1_score, conf_matrix = confusionMatrix(outputs, targets)
    
    # Imprimimos los resultados con un formato claro
    println("====== MÉTRICAS DE EVALUACIÓN ======")
    println("Precisión (Accuracy): \t\t", round(accuracy, digits=4))
    println("Tasa de error: \t\t\t", round(error_rate, digits=4))
    println("Sensibilidad (Recall): \t\t", round(sensitivity, digits=4))
    println("Especificidad: \t\t\t", round(specificity, digits=4))
    println("Valor Predictivo Positivo (VPP):\t", round(precision_val, digits=4))
    println("Valor Predictivo Negativo (VPN):\t", round(NPV, digits=4))
    println("F1-score: \t\t\t", round(f1_score, digits=4))
    println("------------------------------------")
    println("Matriz de Confusión [VN FP; FN VP]:")
    display(conf_matrix)
    println("====================================")
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Convertimos las probabilidades a booleanos usando el umbral y llamamos a la función de impresión anterior
    outputs_bool = outputs .>= threshold
    printConfusionMatrix(outputs_bool, targets)
end

using SymDoME
using GeneticProgramming


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    (trainingInputs, trainingTargets) = trainingDataset
    trainingInputs = Float64.(trainingInputs)
    testInputs     = Float64.(testInputs)

    # Entrena el modelo DoME (clasificación binaria)
    (_, _, _, model) = dome(trainingInputs, trainingTargets; maximumNodes = maximumNodes)

    # Evalúa en test
    testOutputs = evaluateTree(model, testInputs)

    # Si el modelo devuelve un escalar (todos los patrones de la misma clase), lo expandimos
    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testInputs, 1))
    end

    return testOutputs  # Vector Float64
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
   (trainingInputs, trainingTargets) = trainingDataset
    numClasses = size(trainingTargets, 2)

    # Caso binario: una sola columna
    if numClasses == 1
        testOutputs = trainClassDoME((trainingInputs, vec(trainingTargets)), testInputs, maximumNodes)
        return reshape(testOutputs, :, 1)
    end

    # Estrategia uno-contra-todos para más de 2 clases
    @assert(numClasses != 2)
    testOutputsMatrix = zeros(Float64, size(testInputs, 1), numClasses)
    for numClass in 1:numClasses
        testOutputsMatrix[:, numClass] = trainClassDoME(
            (trainingInputs, trainingTargets[:, numClass]),
            testInputs,
            maximumNodes
        )
    end
    return testOutputsMatrix
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    (trainingInputs, trainingTargets) = trainingDataset
    classes = unique(trainingTargets)

    # Vector de salida del mismo tipo que trainingTargets
    testOutputs = Array{eltype(trainingTargets), 1}(undef, size(testInputs, 1))

    # One-hot-encoding con el vector de clases fijo
    testOutputsDoME = trainClassDoME(
        (trainingInputs, oneHotEncoding(trainingTargets, classes)),
        testInputs,
        maximumNodes
    )

    # classifyOutputs con umbral 0 (el signo indica la clase)
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold = 0)

    if length(classes) <= 2
        testOutputsBool = vec(testOutputsBool)
        testOutputs[testOutputsBool]  .= classes[1]
        if length(classes) == 2
            testOutputs[.!testOutputsBool] .= classes[2]
        end
    else
        for numClass in 1:length(classes)
            testOutputs[testOutputsBool[:, numClass]] .= classes[numClass]
        end
    end

    return testOutputs
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    # 1. Vector 1:k, repetido hasta cubrir N, tomar primeros N y desordenar
    indices = shuffle!(repeat(1:k, Int(ceil(N / k)))[1:N])
    return indices
end;


function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = Array{Int64,1}(undef, length(targets))
    # Asignación estratificada para positivos y negativos
    indices[targets]   = crossvalidation(sum(targets), k)
    indices[.!targets] = crossvalidation(sum(.!targets), k)
    return indices
end;


function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    indices = Array{Int64,1}(undef, size(targets, 1))
    for numClass in 1:size(targets, 2)
        mask = targets[:, numClass]
        indices[mask] = crossvalidation(sum(mask), k)
    end
    return indices
end;


function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    return crossvalidation(oneHotEncoding(targets), k)
end;



function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20)

    (inputs, rawTargets) = dataset

    # One-hot-encoding con clases fijas
    classes = unique(rawTargets)
    targets = oneHotEncoding(rawTargets, classes)
    numClasses = length(classes)

    k = maximum(crossValidationIndices)

    # Vectores para métricas por fold
    accValues  = Float64[]
    errValues  = Float64[]
    senValues  = Float64[]
    speValues  = Float64[]
    ppvValues  = Float64[]
    npvValues  = Float64[]
    f1Values   = Float64[]
    confMatrix = zeros(Float64, numClasses, numClasses)

    for fold in 1:k
        # Máscaras de train/test
        testMask  = crossValidationIndices .== fold
        trainMask = .!testMask

        trainInputs  = inputs[trainMask, :]
        trainTargets = targets[trainMask, :]
        testInputs   = inputs[testMask, :]
        testTargets  = targets[testMask, :]

        # Vectores de métricas para las numExecutions ejecuciones de este fold
        accFold = Float64[]; errFold = Float64[]
        senFold = Float64[]; speFold = Float64[]
        ppvFold = Float64[]; npvFold = Float64[]
        f1Fold  = Float64[]
        confFold = zeros(Float64, numClasses, numClasses, numExecutions)

        for exec in 1:numExecutions
            # División opcional en entrenamiento+validación
            if validationRatio > 0
                # El ratio ajustado respecto al fold de entrenamiento
                valRatioAdjusted = validationRatio * size(inputs, 1) / size(trainInputs, 1)
                (tIdx, vIdx) = holdOut(size(trainInputs, 1), valRatioAdjusted)
                (ann, _, _, _) = trainClassANN(topology,
                    (trainInputs[tIdx, :], trainTargets[tIdx, :]);
                    validationDataset = (trainInputs[vIdx, :], trainTargets[vIdx, :]),
                    transferFunctions = transferFunctions,
                    maxEpochs = maxEpochs, minLoss = minLoss,
                    learningRate = learningRate, maxEpochsVal = maxEpochsVal)
            else
                (ann, _, _, _) = trainClassANN(topology,
                    (trainInputs, trainTargets);
                    transferFunctions = transferFunctions,
                    maxEpochs = maxEpochs, minLoss = minLoss,
                    learningRate = learningRate)
            end

            # Evaluación en test
            testOutputs = ann(Float32.(testInputs)')'
            (acc, err, sen, spe, ppv, npv, f1, cm) = confusionMatrix(testOutputs, testTargets)

            push!(accFold, acc); push!(errFold, err)
            push!(senFold, sen); push!(speFold, spe)
            push!(ppvFold, ppv); push!(npvFold, npv)
            push!(f1Fold,  f1)
            confFold[:, :, exec] = Float64.(cm)
        end

        # Promedios del fold
        push!(accValues, mean(accFold)); push!(errValues, mean(errFold))
        push!(senValues, mean(senFold)); push!(speValues, mean(speFold))
        push!(ppvValues, mean(ppvFold)); push!(npvValues, mean(npvFold))
        push!(f1Values,  mean(f1Fold))
        confMatrix .+= mean(confFold, dims=3)[:, :, 1]
    end

    return (
        (mean(accValues), std(accValues)),
        (mean(errValues), std(errValues)),
        (mean(senValues), std(senValues)),
        (mean(speValues), std(speValues)),
        (mean(ppvValues), std(ppvValues)),
        (mean(npvValues), std(npvValues)),
        (mean(f1Values),  std(f1Values)),
        confMatrix
    )
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1})

    (inputs, targets) = dataset

    # Caso ANN: delegar directamente en ANNCrossValidation
    if modelType == :ANN
        return ANNCrossValidation(
            modelHyperparameters["topology"],
            dataset,
            crossValidationIndices;
            numExecutions  = get(modelHyperparameters, "numExecutions",   50),
            maxEpochs      = get(modelHyperparameters, "maxEpochs",       1000),
            minLoss        = get(modelHyperparameters, "minLoss",         0.0),
            learningRate   = get(modelHyperparameters, "learningRate",    0.01),
            validationRatio= get(modelHyperparameters, "validationRatio", 0.0),
            maxEpochsVal   = get(modelHyperparameters, "maxEpochsVal",    20)
        )
    end

    # Para el resto de modelos, convertir targets a String y calcular clases
    targets  = string.(targets)
    classes  = unique(targets)
    k        = maximum(crossValidationIndices)
    numClasses = length(classes)

    accValues = Float64[]; errValues = Float64[]
    senValues = Float64[]; speValues = Float64[]
    ppvValues = Float64[]; npvValues = Float64[]
    f1Values  = Float64[]
    confMatrix = zeros(Float64, numClasses, numClasses)

    for fold in 1:k
        testMask  = crossValidationIndices .== fold
        trainMask = .!testMask

        trainInputs  = inputs[trainMask, :]
        trainTargets = targets[trainMask]
        testInputs   = inputs[testMask,  :]
        testTargets  = targets[testMask]

        if modelType == :DoME
            # DoME: trainClassDoME ya devuelve etiquetas del tipo original
            testOutputs = string.(trainClassDoME(
                (trainInputs, trainTargets), testInputs,
                modelHyperparameters["maximumNodes"]
            ))

        else
            # Construcción del modelo MLJ según el tipo
            if modelType == :SVC
                kernel_str = modelHyperparameters["kernel"]
                kernel_val = if kernel_str == "linear"
                    LIBSVM.Kernel.Linear
                elseif kernel_str == "rbf"
                    LIBSVM.Kernel.RadialBasis
                elseif kernel_str == "sigmoid"
                    LIBSVM.Kernel.Sigmoid
                elseif kernel_str == "poly"
                    LIBSVM.Kernel.Polynomial
                end
                model = SVMClassifier(
                    kernel = kernel_val,
                    cost   = Float64(modelHyperparameters["C"]),
                    gamma  = Float64(get(modelHyperparameters, "gamma",  1.0)),
                    degree = Int32(  get(modelHyperparameters, "degree", 3)),
                    coef0  = Float64(get(modelHyperparameters, "coef0",  0.0))
                )

            elseif modelType == :DecisionTreeClassifier
                model = DTClassifier(
                    max_depth = get(modelHyperparameters, "max_depth", -1),
                    rng       = Random.MersenneTwister(1)
                )

            elseif modelType == :KNeighborsClassifier
                model = kNNClassifier(
                    K = modelHyperparameters["n_neighbors"]
                )

            else
                error("Tipo de modelo desconocido: $modelType")
            end

            # Entrenar con MLJ
            mach = machine(model,
                MLJ.table(trainInputs),
                categorical(trainTargets; levels = classes))
            MLJ.fit!(mach, verbosity=0)

            # Predecir
            if modelType == :SVC
                testOutputs = string.(MLJ.predict(mach, MLJ.table(testInputs)))
            else
                # kNN y DecisionTree devuelven distribución → aplicar mode
                testOutputs = string.(mode.(MLJ.predict(mach, MLJ.table(testInputs))))
            end
        end

        # Calcular métricas (siempre con el vector de clases explícito)
        (acc, err, sen, spe, ppv, npv, f1, cm) =
            confusionMatrix(testOutputs, testTargets, classes)

        push!(accValues, acc); push!(errValues, err)
        push!(senValues, sen); push!(speValues, spe)
        push!(ppvValues, ppv); push!(npvValues, npv)
        push!(f1Values,  f1)
        confMatrix .+= Float64.(cm)
    end

    return (
        (mean(accValues), std(accValues)),
        (mean(errValues), std(errValues)),
        (mean(senValues), std(senValues)),
        (mean(speValues), std(speValues)),
        (mean(ppvValues), std(ppvValues)),
        (mean(npvValues), std(npvValues)),
        (mean(f1Values),  std(f1Values)),
        confMatrix
    )
end;

