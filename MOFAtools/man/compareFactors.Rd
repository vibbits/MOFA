% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/compareModels.R
\name{compareFactors}
\alias{compareFactors}
\title{Plot the robustness of the latent factors across diferent trials}
\usage{
compareFactors(models, comparison = "all", show_rownames = FALSE,
  show_colnames = FALSE, ...)
}
\arguments{
\item{models}{a list containing \code{\link{MOFAmodel}} objects.}

\item{comparison}{tye of comparison, either 'pairwise' or 'all'}

\item{...}{extra arguments passed to pheatmap}
}
\value{
Plots a heatmap of correlation of Latent Factors in all models when 'comparison' is 'all'. 
Otherwise, for each pair of models, a seperate heatmap is produced comparing one model againt the other.
The corresponding correlation matrix or list or pairwise correlation matrices is returned
}
\description{
Different objects of \code{\link{MOFAmodel}} are compared in terms of correlation between 
their latent factors. The correlation is calculated only on those samples which are present in all models.
Ideally, the output should look like a block diagonal matrix, suggesting that all detected factors are robust under different initialisations.
If not, it suggests that some factors are weak and not captured by all models.
}
\details{
TO-FILL
}
\references{
fill this
}
