
##############################################
## Functions to visualise the training data ##
##############################################

#' @title Plot heatmap of relevant features
#' @name plotDataHeatmap
#' @description Function to plot a heatmap of the input data for relevant features, 
#' usually the ones with highest loadings in a given factor.
#' @param object a \code{\link{MOFAmodel}} object.
#' @param view character vector with the view name, or numeric vector with the index of the view.
#' @param factor character vector with the factor name, or numeric vector with the index of the factor.
#' @param features if an integer, the total number of top features to plot, based on the absolute value of the loading.
#' If a character vector, a set of manually-defined features. 
#' Default is 50.
#' @param includeWeights logical indicating whether to include the weight of each feature as an extra annotation in the heatmap. 
#' Default is FALSE.
#' @param transpose logical indicating whether to transpose the output heatmap. 
#' Default corresponds to features as rows and samples as columns.
#' @param imputed logical indicating whether to plot the imputed data instead of the original data. 
#' Default is FALSE.
#' @param ... further arguments that can be passed to \code{\link[pheatmap]{pheatmap}}
#' @details One of the first steps for the annotation of a given factor is to visualise the corresponding loadings, 
#' using for example \code{\link{plotWeights}} or \code{\link{plotTopWeights}}.
#' These functions display the top features that are driving the heterogeneity captured by a factor. \cr
#' However, one might also be interested in visualising the coordinated heterogeneity in the input data, 
#' rather than looking at "abstract" weights. \cr
#' This function extracts the top features for a given factor and view, 
#' and generates a heatmap with dimensions (samples,features). This should reveal
#' the underlying heterogeneity that is captured by the latent factor. \cr
#' A similar function for doing scatterplots rather than heatmaps is \code{\link{plotDataScatter}}.
#' @import pheatmap
#' @examples
#' # Load CLL data
#' filepath <- system.file("extdata", "CLL_model.hdf5", package = "MOFAtools")
#' MOFA_CLL <- loadModel(filepath)
#' # plot top 30 features on Factor 1 in the mRNA view
#' plotDataHeatmap(MOFA_CLL, view="mRNA", factor=1, features=30)
#' # without column names (extra arguments passed to pheatmap)
#' plotDataHeatmap(MOFA_CLL, view="mRNA", factor=1, features=30, show_colnames = FALSE)
#' # transpose the heatmap
#' plotDataHeatmap(MOFA_CLL, view="mRNA", factor=1, features=30, transpose=TRUE)
#' # do not cluster rows (extra arguments passed to pheatmap)
#' plotDataHeatmap(MOFA_CLL, view="mRNA", factor=1, features=30, cluster_rows=FALSE)
#' @export
plotDataHeatmap <- function(object, view, factor, features = 50, includeWeights = FALSE, 
                            transpose = FALSE, imputed = FALSE, ...) {
  
  # Sanity checks
  if (!is(object, "MOFAmodel")) stop("'object' has to be an instance of MOFAmodel")
  
  # Get views
  if (is.numeric(view)) view <- viewNames(object)[view]
  stopifnot(view %in% viewNames(object))
  stopifnot(length(factor)==1)

  # Get factors
  if (is.numeric(factor)) {
    if (object@ModelOptions$learnIntercept == T) {
      factor <- factorNames(object)[factor+1]
    } else {
      factor <- factorNames(object)[factor]
    }
  } else { 
    stopifnot(factor %in% factorNames(object)) 
  }

  # Collect relevant data
  W <- getWeights(object)[[view]][,factor]
  Z <- getFactors(object)[,factor]
  Z <- Z[!is.na(Z)]
  
  if (imputed==TRUE) {
    data <- getImputedData(object, view)[[1]][,names(Z)]
  } else {
    data <- getTrainData(object, view)[[1]][,names(Z)]
  }
  
  # Ignore samples with full missing views
  data <- data[,apply(data, 2, function(x) !all(is.na(x)))]
  
  # Define features
  if (class(features) == "numeric") {
    features <- names(tail(sort(abs(W)), n=features))
    stopifnot(all(features %in% featureNames(object)[[view]]))
  } else if (class(features)=="character") {
    stopifnot(all(features %in% featureNames(object)[[view]]))
  } else {
    stop("Features need to be either a numeric or character vector")
  }
  data <- data[features,]
  
  # Sort samples according to the latent factor
  order_samples <- names(sort(Z, decreasing=T))
  order_samples <- order_samples[order_samples %in% colnames(data)]
  data <- data[,order_samples]
  
  # Transpose the data
  if (transpose==T) { data <- t(data) }
  
  # Plot heatmap
  if (includeWeights==TRUE) { 
    anno <- data.frame(row.names=names(W[features]), weight=W[features]) 
    if (transpose==T) {
      pheatmap(t(data), annotation_col=anno, ...)
    } else {
      pheatmap(t(data), annotation_row=anno, ...)
    }
  } else {
    pheatmap(t(data), ...)
  }
  
}



#' @title Scatterplots of feature values against latent factors
#' @name plotDataScatter
#' @description Function to do a scatterplot of the feature(s) values against the latent factor values.
#' @param object a \code{\link{MOFAmodel}} object.
#' @param view character vector with a view name, or numeric vector with the index of the view.
#' @param factor character vector with a factor name, or numeric vector with the index of the factor.
#' @param features if an integer, the total number of features to plot (10 by default). If a character vector, a set of manually-defined features.
#' @param color_by specifies groups or values used to color the samples. 
#' This can be either: 
#' (a) a character giving the name of a feature, 
#' (b) a character giving the same of a covariate (only if using MultiAssayExperiment as input), or
#' (c) a vector of the same length as the number of samples specifying discrete groups or continuous numeric values.
#' @param shape_by specifies groups or values used to shape the samples. 
#' This can be either: 
#' (a) a character giving the name of a feature present in the training data, 
#' (b) a character giving the same of a covariate (only if using MultiAssayExperiment as input), or 
#' (c) a vector of the same length as the number of samples specifying discrete groups.
#' @param name_color name for the color legend
#' @param name_shape name for the shape legend
#' @param showMissing logical indicating whether to show samples with missing values for the color or the shape.
#' Default is TRUE.
#' @details One of the first steps for the annotation of a given factor is to visualise the corresponding loadings, 
#' using for example \code{\link{plotWeights}} or \code{\link{plotTopWeights}}.
#' These functions display the top features that are driving the heterogeneity captured by a factor. \cr
#' However, one might also be interested in visualising the coordinated heterogeneity in the input data, 
#' rather than looking at "abstract" weights. \cr
#' This function generates scatterplots of features against factors (each dot is a sample), 
#' so that you can observe the association between them. \cr
#' A similar function for doing heatmaps rather than scatterplots is \code{\link{plotDataHeatmap}}.
#' @import ggplot2
#' @import dplyr
#' @export
#' @examples
#' # Load CLL data
#' filepath <- system.file("extdata", "CLL_model.hdf5", package = "MOFAtools")
#' MOFA_CLL <- loadModel(filepath)
#' # plot scatter for top 5 features on factor 1 in the view mRNA:
#' plotDataScatter(MOFA_CLL, view="mRNA", factor=1, features=5)
#' # coloring by the IGHV status (features in Mutations view), not showing samples with missing IGHV:
#' plotDataScatter(MOFA_CLL, view="mRNA", factor=1, features=5, color_by="IGHV", showMissing=FALSE)
plotDataScatter <- function(object, view, factor, features = 10,
                            color_by=NULL, name_color="",  
                            shape_by=NULL, name_shape="", showMissing = TRUE) {
  
  # Sanity checks
  if (!is(object, "MOFAmodel")) stop("'object' has to be an instance of MOFAmodel")
  stopifnot(length(factor)==1)
  stopifnot(length(view)==1)
  if (!view %in% viewNames(object)) stop(sprintf("The view %s is not present in the object",view))

  if(is.numeric(factor)) {
      if (object@ModelOptions$learnIntercept) factor <- factorNames(object)[factor+1]
      else factor <- factorNames(object)[factor]
    } else{ stopifnot(factor %in% factorNames(object)) }
      
  # Collect relevant data
  N <- getDimensions(object)[["N"]]
  Z <- getFactors(object)[,factor]
  W <- getWeights(object, views=view, factors=factor)[[1]][,1]
  Y <- object@TrainData[[view]]
  
  # Get features
  if (class(features) == "numeric") {
    features <- names(tail(sort(abs(W)), n=features))
    stopifnot(all(features %in% featureNames(object)[[view]]))
  } else if (class(features)=="character") {
    stopifnot(all(features %in% featureNames(object)[[view]]))
  } else {
    stop("Features need to be either a numeric or character vector")
  }
  W <- W[features]
  Y <- Y[features,]
  
  
  # Set color
  colorLegend <- T
  if (!is.null(color_by)) {
    # It is the name of a covariate or a feature in the TrainData
    if (length(color_by) == 1 & is.character(color_by)) {
      if(name_color=="") name_color <- color_by
      TrainData <- getTrainData(object)
      featureNames <- lapply(TrainData(object), rownames)
      if(color_by %in% Reduce(union,featureNames)) {
        viewidx <- which(sapply(featureNames, function(vnm) color_by %in% vnm))
        color_by <- TrainData[[viewidx]][color_by,]
      } else if(class(object@InputData) == "MultiAssayExperiment"){
        color_by <- getCovariates(object, color_by)
      }
      else stop("'color_by' was specified but it was not recognised, please read the documentation")
      # It is a vector of length N
    } else if (length(color_by) > 1) {
      stopifnot(length(color_by) == N)
      # color_by <- as.factor(color_by)
    } else {
      stop("'color_by' was specified but it was not recognised, please read the documentation")
    }
  } else {
    color_by <- rep(TRUE,N)
    colorLegend <- F
  }

  # Set shape
  shapeLegend <- T
  if (!is.null(shape_by)) {
    # It is the name of a covariate 
    if (length(shape_by) == 1 & is.character(shape_by)) {
      if(name_shape=="") name_shape <- shape_by
      TrainData <- getTrainData(object)
      featureNames <- lapply(TrainData(object), rownames)
      if (shape_by %in% Reduce(union,featureNames)) {
        viewidx <- which(sapply(featureNames, function(vnm) shape_by %in% vnm))
        shape_by <- TrainData[[viewidx]][shape_by,]
      } else if(class(object@InputData) == "MultiAssayExperiment"){
        shape_by <- getCovariates(object, shape_by)
      }
      else stop("'shape_by' was specified but it was not recognised, please read the documentation")
      # It is a vector of length N
      # It is a vector of length N
    } else if (length(shape_by) > 1) {
      stopifnot(length(shape_by) == N)
    } else {
      stop("'shape_by' was specified but it was not recognised, please read the documentation")
    }
  } else {
    shape_by <- rep(TRUE,N)
    shapeLegend <- F
  }
  
  
  # Create data frame 
  df1 <- data.frame(sample=names(Z), x = Z, shape_by = shape_by, color_by = color_by, stringsAsFactors=F)
  df2 <- getTrainData(object, views=view, features = list(features), as.data.frame=T)
  df <- dplyr::left_join(df1,df2, by="sample")
    # Remove samples with missing values
  if (!showMissing) {
    df <- df[!(is.na(df$shape_by) | is.na(df$color_by)),]  
  }
  if(length(unique(df$color_by)) < 5) df$color_by <- as.factor(df$color_by)

  # Generate plot
  p <- ggplot(df, aes_string(x = "x", y = "value", color = "color_by", shape = "shape_by")) + 
    geom_point() +
    xlab(paste0("Latent factor ", factor)) + ylab("Feature value") +
    # ggbeeswarm::geom_quasirandom() +
    stat_smooth(method="lm", color="blue", alpha=0.5) +
    facet_wrap(~feature, scales="free_y") +
    scale_shape_manual(values=c(19,1,2:18)[1:length(unique(shape_by))]) +
    theme(plot.margin = margin(20, 20, 10, 10), 
          axis.text = element_text(size = rel(1), color = "black"), 
          axis.title = element_text(size = 16), 
          axis.title.y = element_text(size = rel(1.1), margin = margin(0, 15, 0, 0)), 
          axis.title.x = element_text(size = rel(1.1), margin = margin(15, 0, 0, 0)), 
          axis.line = element_line(color = "black", size = 0.5), 
          axis.ticks = element_line(color = "black", size = 0.5),
          panel.border = element_blank(), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(), 
          panel.background = element_blank(),
          legend.key = element_rect(fill = "white")
          # legend.text = element_text(size = titlesize),
          # legend.title = element_text(size =titlesize)
          )
  if (colorLegend) { p <- p + labs(color = name_color) } else { p <- p + guides(color = FALSE) }
  if (shapeLegend) { p <- p + labs(shape = name_shape) }  else { p <- p + guides(shape = FALSE) }
  
  return(p)
}



#' @title Tile plot of the multi-omics data
#' @name plotTilesData
#' @description Function to do a tile plot showing the dimensionality and 
#' the missing value structure of the multi-omics data.
#' @param object a \code{\link{MOFAmodel}} object.
#' @param colors a character vector specifying the colors per view. NULL (default) uses an internal palette.
#' @details This function is helpful to get an overview of the dimensionality and the missing value structure of the training data. \cr 
#' It shows the number of samples, the number of views, the number of features, and the structure of missing values. \cr
#' It is particularly useful to visualise incomplete data sets, where some samples are missing subsets of assays.
#' @import ggplot2
#' @import dplyr
#' @import reshape2
#' @export
#' @examples
#' # Example on the CLL data
#' filepath <- system.file("extdata", "CLL_model.hdf5", package = "MOFAtools")
#' MOFA_CLL <- loadModel(filepath)
#' plotTilesData(MOFA_CLL)
#'
#' # Example on the scMT data
#' filepath <- system.file("extdata", "scMT_model.hdf5", package = "MOFAtools")
#' MOFA_scMT <- loadModel(filepath)
#' # using customized colors
#' plotTilesData(MOFA_scMT, colors= c("blue", "red", "red", "red"))

plotTilesData <- function(object, colors = NULL) {
  
  # Sanity checks
  if (!is(object, "MOFAmodel")) stop("'object' has to be an instance of MOFAmodel")
  
  # Collect relevant data
  TrainData <- object@TrainData
  M <- getDimensions(object)[["M"]]
  
  # Define colors  
  if (is.null(colors)) {
    palette <- c("#D95F02", "#377EB8", "#E6AB02", "#31A354", "#7570B3", "#E7298A", "#66A61E",
                 "#A6761D", "#666666", "#E41A1C", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33",
                 "#A65628", "#F781BF", "#1B9E77")
    if (M<17) colors <- palette[1:M] else colors <- rainbow(M)
  }
  if (length(colors)!=M) stop("Length of 'colors' does not match the number of views")
  names(colors) <- viewNames(object)
  
  # Define availability binary matrix to indicate whether assay j is profiled in sample i
  ovw <- sapply(TrainData, function(dat) apply(dat,2,function(s) !all(is.na(s))))
  
  # Remove samples with no measurements
  ovw <- ovw[apply(ovw,1,any),, drop=FALSE]
  
  # Melt to data.frame
  molten_ovw <- melt(ovw, varnames=c("sample", "view"))
  
  # order samples
  molten_ovw$sample <- factor(molten_ovw$sample, levels = rownames(ovw)[order(rowSums(ovw), decreasing = T)])
  n <- length(unique(molten_ovw$sample))
  
  # Add number of samples and features per view
  molten_ovw$combi <- ifelse(molten_ovw$value, as.character(molten_ovw$view), "missing")
  molten_ovw$ntotal <- paste("n=", colSums(ovw)[as.character(molten_ovw$view) ], sep="")
  molten_ovw$ptotal <- paste("d=", sapply(TrainData, nrow)[as.character(molten_ovw$view) ], sep="")
    
  # Define y-axis label
  molten_ovw <-  mutate(molten_ovw, view_label = paste(view, ptotal, sep="\n"))
  
  # Plot
  p <- ggplot(molten_ovw, aes(x=sample, y=view_label, fill=combi)) +
    geom_tile(width=0.7, height=0.9, col="black") +
    geom_text(data=filter(molten_ovw, sample==levels(molten_ovw$sample)[1]),
              aes(x=levels(molten_ovw$sample)[n/2],label=ntotal), size=6) +
    scale_fill_manual(values = c('missing'="grey", colors)) +
    # ggtitle("Samples available for training") +
    xlab(paste0("Samples (n=",n,")")) + ylab("") +
    guides(fill=F) + 
    theme(
      axis.text.x =element_blank(),
      panel.background = element_rect(fill="white"),
      text = element_text(size=16),
      axis.ticks.y = element_blank(),
      axis.ticks.x= element_blank(),
      axis.text.y = element_text(color="black"),
      panel.grid = element_line(colour = "black"),
      plot.margin = unit(c(5.5,2,5.5,5.5), "pt")
    ) 
  
  return(p)
}

