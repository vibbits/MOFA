library(corrplot)

#Function to correlate Latent factor to principal components of specified view
CorrplotLFvsPC<-function(modelobject, viewname4PC, noPCs=5){
  Z<-modelobject@Expectations$Z$E
  colnames(Z)<-paste("LF", 1:ncol(Z), sep="")
  singleview<-modelobject@Data[[viewname4PC]]
  pc.out<-prcomp(singleview)
  
  corrmatrix<-apply(pc.out$x[,1:noPCs],2, function(pc) {
    apply(Z,2, function(lv){
      cor(pc, lv)
    })
  })
  
  corrplot(corrmatrix, order="original", title=viewname4PC,mar = c(1, 1, 3, 1))

}