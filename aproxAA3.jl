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
    carpetas_dataset = ["No-tumor", "Tumor"]

    for carpeta in carpetas_dataset
        subdir = joinpath(base_path, carpeta) 
        if !isdir(subdir)
            error("Directorio no encontrado: $subdir. Asegúrate de tener las carpetas ahí.")
        end
        
        # Etiqueta binaria
        etiqueta_binaria = (carpeta == "No-tumor") ? "sano" : "Tumor"

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
    println("    Tumores: ", count(x -> x == "Tumor", labels))
    
    return images, labels
end

println("Iniciando carga de datos...")
images, labels = read_images_and_labels("./datasets") 
num_images = length(images)
println("Cargadas $num_images imagenes correctamente.")

# 3. Extraer características (11 características específicas)

# Función de entropía existente
function hist_intensidades(data; bins=256)
    h = fit(Histogram, data, range(0, stop=1, length=bins))
    prob = h.weights ./ sum(h.weights)
    return filter(x -> x > 0, prob)
end

function entropy(p)
    return -sum(p .* log2.(p))
end

# Función para calcular características GLCM (Textura)
function calc_glcm_features(img_2d; levels=16)
    # Cuantizar la imagen a un número menor de niveles de gris (1 a 16)
    quantized = round.(Int, img_2d .* (levels - 1)) .+ 1
    glcm = zeros(Float64, levels, levels)
    rows, cols = size(quantized)
    
    # Rellenar la matriz de co-ocurrencia (relación horizontal, distancia 1)
    for r in 1:rows, c in 1:(cols-1)
        i = quantized[r, c]
        j = quantized[r, c+1]
        glcm[i, j] += 1.0
        glcm[j, i] += 1.0 # Hacerla simétrica
    end
    
    sum_glcm = sum(glcm)
    sum_glcm == 0 && return (0.0, 0.0, 0.0, 0.0)
    p = glcm ./ sum_glcm

    contrast, homogeneity, energy, correlation = 0.0, 0.0, 0.0, 0.0
    
    # Calcular medias y varianzas para la correlación
    mu_i = sum(i * p[i,j] for i in 1:levels, j in 1:levels)
    mu_j = sum(j * p[i,j] for i in 1:levels, j in 1:levels)
    var_i = sum((i - mu_i)^2 * p[i,j] for i in 1:levels, j in 1:levels)
    var_j = sum((j - mu_j)^2 * p[i,j] for i in 1:levels, j in 1:levels)
    sigma_i, sigma_j = sqrt(var_i), sqrt(var_j)

    for i in 1:levels, j in 1:levels
        contrast += (i - j)^2 * p[i, j]
        homogeneity += p[i, j] / (1.0 + abs(i - j))
        energy += p[i, j]^2
        if sigma_i > 0 && sigma_j > 0
            correlation += ((i - mu_i)*(j - mu_j)*p[i, j]) / (sigma_i * sigma_j)
        end
    end
    
    return contrast, homogeneity, energy, correlation
end

# Función para calcular características Morfológicas (Área y Solidez)
function calc_morphology(img_2d)
    # Binarizar la imagen para separar "la masa" del fondo
    # Usamos un umbral basado en la media y desviación (suele resaltar tumores brillantes)
    umbral = mean(img_2d) + 0.5 * std(img_2d)
    binary = img_2d .> umbral
    
    area = sum(binary)
    
    # Calcular Solidez (Proporción respecto a su Bounding Box)
    active_pixels = findall(binary)
    if isempty(active_pixels)
        return 0.0, 0.0
    end
    
    rs = [p[1] for p in active_pixels]
    cs = [p[2] for p in active_pixels]
    bbox_area = (maximum(rs) - minimum(rs) + 1) * (maximum(cs) - minimum(cs) + 1)
    
    solidez = bbox_area > 0 ? area / bbox_area : 0.0
    
    return float(area), float(solidez)
end

# Bucle principal de extracción
function calculate_image_statistics(images)
    n = length(images)
    results = zeros(n, 11) # Matriz para 11 características
    
    for (i, img) in enumerate(images)
        # Convertir HxWxC a 2D (gris) sin asumir que la primera dimensión sea 1.
        if size(img, 3) == 1
            img_2d = img[:, :, 1]
        elseif size(img, 3) >= 3
            img_2d = clamp.(0.2989 .* img[:, :, 1] .+ 0.5870 .* img[:, :, 2] .+ 0.1140 .* img[:, :, 3], 0.0, 1.0)
        else
            img_2d = dropdims(mean(img, dims=3), dims=3)
        end
        data = vec(img_2d) # Vector 1D para estadísticas globales
        
        # 1-3. Estadísticas Globales
        results[i, 1] = mean(data)                                # 1. Media
        results[i, 2] = std(data)                                 # 2. Desviación Típica
        results[i, 3] = entropy(hist_intensidades(data))          # 3. Entropía
        
        # 4-7. Características GLCM (Textura)
        contrast, homog, energy, correl = calc_glcm_features(img_2d)
        results[i, 4] = contrast                                  # 4. Contraste
        results[i, 5] = homog                                     # 5. Homogeneidad
        results[i, 6] = energy                                    # 6. Energía
        results[i, 7] = correl                                    # 7. Correlación
        
        # 8-9. Características Morfológicas
        area, solidez = calc_morphology(img_2d)
        results[i, 8] = area                                      # 8. Área de la masa
        results[i, 9] = solidez                                   # 9. Solidez / Compacidad
        
        # 10-11. Límites de intensidad
        results[i, 10] = maximum(data)                            # 10. Máximo
        results[i, 11] = results[i, 10] - minimum(data)           # 11. Rango dinámico
    end
    return results
end

println("Extrayendo 11 características personalizadas por imagen...")
stats = calculate_image_statistics(images)

function zscore_features(data::AbstractArray{<:Real,2})
    mu = mean(data, dims=1)
    sigma = std(data, dims=1)
    sigma_safe = copy(sigma)
    sigma_safe[sigma_safe .== 0] .= 1.0
    return (data .- mu) ./ sigma_safe
end

stats = zscore_features(stats)
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
    config = Dict("kernel" => kernel, "C" => C, "tolerance" => 0.01, "shrinking" => true)
    if kernel == "poly"
        config["degree"] = 3
        config["coef0"] = 1.0
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

# 7. Mostrar resultados y gráficas — 5 gráficas de líneas, una por métrica
# Métricas que se graficarán (puedes añadir o quitar según convenga)
metricsToPlot = ["Accuracy", "Recall", "Specif.", "Precision", "F1"]

function plot_results(df::DataFrame; metrics::Vector{String}=metricsToPlot)
    modelos   = unique(df.model)
    # Paleta de colores — uno por modelo
    colores   = [:royalblue, :crimson, :forestgreen, :darkorange, :purple]

    for metric in metrics
        df_metric = filter(row -> row.metric == metric, df)

        # Imprimir tabla resumen
        println("\n=== Métrica: $metric ===")
        for mod in modelos
            df_mod = filter(r -> r.model == mod, df_metric)
            # Una fila por configuración (params), ordenada tal como aparece
            configs = unique(df_mod.params)
            for (idx, cfg) in enumerate(configs)
                row_cfg = filter(r -> r.params == cfg, df_mod)
                if nrow(row_cfg) > 0
                    println("  [$mod | cfg $idx] Media: $(round(row_cfg[1,:mean_test], digits=4))  STD: $(round(row_cfg[1,:std_test], digits=4))  Params: $cfg")
                end
            end
        end

        # Construir la figura de líneas
        plt = plot(
            title  = "Comparativa de configuraciones — $metric",
            xlabel = "Índice de configuración",
            ylabel = metric,
            legend = :outertopright,
            framestyle = :box,
            size   = (900, 500),
            margin = 8Plots.mm
        )

        for (i, mod) in enumerate(modelos)
            df_mod = filter(r -> r.model == mod, df_metric)
            configs = unique(df_mod.params)
            medias  = Float64[]
            stds    = Float64[]

            for cfg in configs
                fila = filter(r -> r.params == cfg, df_mod)
                if nrow(fila) > 0
                    push!(medias, fila[1, :mean_test])
                    push!(stds,   fila[1, :std_test])
                end
            end

            x = 1:length(medias)
            color = colores[mod(i - 1, length(colores)) + 1]

            plot!(plt, x, medias,
                label      = mod,
                color      = color,
                linewidth  = 2,
                marker     = :circle,
                markersize = 4
            )
            # Banda de desviación típica (±1 std)
            plot!(plt, x, medias .+ stds,
                label     = "",
                color     = color,
                linewidth = 0.5,
                linestyle = :dash,
                alpha     = 0.5
            )
            plot!(plt, x, medias .- stds,
                label     = "",
                color     = color,
                linewidth = 0.5,
                linestyle = :dash,
                alpha     = 0.5
            )
        end

        display(plt)
        savefig(plt, "grafica_$(replace(metric, "." => ""))_lineas.png")
        println("Guardada: grafica_$(replace(metric, "." => ""))_lineas.png")
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
plot_comparison_best_f1_per_model(results)  # Mantiene la gráfica de barras del mejor F1 por modelo