using Pkg
# Añadido StatsBase a la lista de paquetes
Pkg.add(["Images", "FileIO", "ImageIO", "ImageMagick", "Statistics", "StatsBase", "MLJ" ,"LIBSVM", "MLJLIBSVMInterface" ,"NearestNeighborModels", "MLJDecisionTreeInterface", "DataFrames", "Plots"])

using Images
using FileIO
using ImageIO
using ImageMagick
using Statistics
using StatsBase # Importante para skewness, kurtosis e histogramas
using Plots
using Random
using MLJ
using DataFrames

# 1. Fijar la semilla aleatoria para garantizar repetibilidad
Random.seed!(123)

# Cargar las funciones de la práctica 1
include("./fonts/firmas.jl")

# 2. Cargar los datos y agrupar en problema BINARIO (Tumor vs Sano)
function read_images_and_labels(base_path::String)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"]
    images = Vector{Array{Float64,3}}()
    labels = String[]
    
    # Las 2 carpetas originales
    carpetas_dataset = ["no-tumor", "tumor"]

    for carpeta in carpetas_dataset
        subdir = joinpath(base_path, carpeta) 
        if !isdir(subdir)
            error("Directorio no encontrado: $subdir. Asegúrate de tener las carpetas ahí.")
        end
        
        # Etiqueta binaria
        etiqueta_binaria = (carpeta == "no-tumor") ? "sano" : "tumor"

        for file in readdir(subdir, join=true)
            if any(ext -> endswith(lowercase(file), ext), image_extensions)
                img = FileIO.load(file)
                img_array = Float64.(channelview(img))
                
                if ndims(img_array) == 2
                    img_array = reshape(img_array, 1, size(img_array)...)
                end
                
                img_array = permutedims(img_array, (2, 3, 1))
                push!(images, img_array)
                push!(labels, etiqueta_binaria)
            end
        end
    end
    
    # Comprobación de seguridad por pantalla
    println("--> Recuento de clases:")
    println("    Sanos: ", count(x -> x == "sano", labels))
    println("    Tumores: ", count(x -> x == "tumor", labels))
    
    return images, labels
end

println("Iniciando carga de datos...")
images, labels = read_images_and_labels("./datasets") 
num_images = length(images)
println("Cargadas $num_images imagenes correctamente.")

# 3. Extraer características (Ahora 10 características estadísticas)

# Funciones auxiliares para calcular la entropía del histograma
function hist_intensidades(data; bins=256)
    h = fit(Histogram, data, range(0, stop=1, length=bins))
    prob = h.weights ./ sum(h.weights)
    return filter(x -> x > 0, prob) # Filtrar ceros para evitar log(0)
end

function entropy(p)
    return -sum(p .* log2.(p))
end

function calculate_image_statistics(images)
    n = length(images)
    results = zeros(n, 10) # Matriz ampliada a 10 columnas
    
    for (i, img) in enumerate(images)
        data = vec(img) # Convertir la imagen 3D a un vector plano de píxeles
        
        results[i, 1] = mean(data)               # 1. Media
        results[i, 2] = var(data)                # 2. Varianza
        results[i, 3] = std(data)                # 3. Desviación Típica
        results[i, 4] = skewness(data)           # 4. Sesgo (Skewness)
        results[i, 5] = kurtosis(data)           # 5. Curtosis
        results[i, 6] = entropy(hist_intensidades(data)) # 6. Entropía
        results[i, 7] = mean(data.^2)            # 7. Energía
        results[i, 8] = maximum(data)            # 8. Máximo
        results[i, 9] = minimum(data)            # 9. Mínimo
        results[i, 10] = results[i, 8] - results[i, 9] # 10. Rango Dinámico
    end
    return results
end

println("Extrayendo 10 características estadísticas por imagen...")
stats = calculate_image_statistics(images)
dataset = (stats, labels)

# 4. Generar índices de validación cruzada (5 folds)
rng = MersenneTwister(123)
crossValidationIndices = rand(rng, 1:5, num_images)

# 5. Configuración de modelos [cite: 116, 117, 119, 121, 122]
modelConfigs = []

# RR.NN.AA (8 arquitecturas)
ann_topologies = [[5], [10], [20], [50], [10, 5], [20, 10], [50, 20], [100, 50]]
for topology in ann_topologies
    push!(modelConfigs, (:ANN, Dict("topology" => topology, "numExecutions" => 10, "maxEpochs" => 500, "minLoss" => 0.0, "learningRate" => 0.01)))
end

# SVM (16 configuraciones)
svc_kernels = ["linear", "poly", "rbf", "sigmoid"]
svc_C = [0.1, 1.0, 10.0, 100.0]
for kernel in svc_kernels, C in svc_C
    config = Dict("kernel" => kernel, "C" => C)
    if kernel == "poly"
        config["degree"] = 3
        config["gamma"] = 0.0
        config["coef0"] = 0.0
    end
    push!(modelConfigs, (:SVC, config))
end

# Árboles de Decisión (6 profundidades)
for d in [3, 5, 7, 10, 15, 20]
    push!(modelConfigs, (:DecisionTreeClassifier, Dict("max_depth" => d)))
end

# k-Nearest Neighbors (6 valores de k)
for k in [1, 3, 5, 7, 9, 11]
    push!(modelConfigs, (:KNeighborsClassifier, Dict("n_neighbors" => k)))
end

# DoME (8 valores de número de nodos)
for nodes in [50, 100, 150, 200, 250, 300, 350, 400]
    push!(modelConfigs, (:DoME, Dict("maximumNodes" => nodes)))
end

# 6. Ejecutar modelos y almacenar resultados
println("Iniciando validación cruzada y entrenamiento de modelos...")
results = DataFrame(model=String[], params=String[], metric=String[], mean_test=Float64[], std_test=Float64[])
metricsNames = ["Accuracy", "ErrorRate", "Recall", "Specif.", "Precision", "NPV", "F1"]

for (modelType, hyperparams) in modelConfigs
    cv = modelCrossValidation(modelType, hyperparams, dataset, crossValidationIndices)
    for (i, mname) in enumerate(metricsNames)
        push!(results, (string(modelType), string(hyperparams), mname, cv[i][1], cv[i][2]))
    end
end
println("Entrenamiento finalizado. Resultados recopilados: ", nrow(results), " filas")

# 7. Mostrar resultados y gráficas [cite: 36]
function plot_results(df::DataFrame)
    modelos = unique(df.model)
    for mod in modelos
        dfm = filter(row -> row.model == mod, df)
        
        agg = combine(groupby(dfm, :metric), :mean_test => mean => :mean)
        métricas = agg.metric
        valores = agg.mean

        dfm_agg = combine(groupby(dfm, [:params, :metric]),
            :mean_test => mean => :mean,
            :std_test => mean => :std)

        println("\n--- Resultados Detallados para $mod ---")
        for row in eachrow(dfm_agg)
            println("Params: $(row.params) | Metrica: $(row.metric) | Media: $(round(row.mean, digits=4)) | STD: $(round(row.std, digits=4))")
        end

        plt = bar(
            métricas, valores,
            xlabel="Métrica", ylabel="Valor medio",
            title="Desempeño TEST - $mod\n(Promedio de configuraciones)",
            legend=false, rotation=45, framestyle=:box
        )
        display(plt)
    end
end

function plot_comparison_best_f1_per_model(df::DataFrame)
    println("\n=== Comparación F1 de la mejor configuración por modelo ===")
    modelos = String[]
    f1_scores = Float64[]

    for mod in unique(df.model)
        df_mod = filter(r -> r.model == mod && r.metric == "F1", df)
        if nrow(df_mod) > 0
            best_row = sort(df_mod, :mean_test, rev=true)[1, :]
            push!(modelos, mod)
            push!(f1_scores, best_row.mean_test)
            println("Modelo: $mod | Mejor F1: $(round(best_row.mean_test, digits=4)) | Config: $(best_row.params)")
        end
    end

    plt = bar(
        modelos, f1_scores,
        xlabel="Modelo", ylabel="Mejor F1-score",
        title="Comparativa final de modelos (Mejor F1)",
        legend=false, framestyle=:box, bar_width=0.5
    )
    display(plt)
end

plot_results(results)
plot_comparison_best_f1_per_model(results)
