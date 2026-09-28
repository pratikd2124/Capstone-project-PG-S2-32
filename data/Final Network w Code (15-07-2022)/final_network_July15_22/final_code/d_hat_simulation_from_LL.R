# SPECIFICATIONS -------------------------------------------------------

testing=TRUE

is_local = TRUE

num_exps = 1000
num_trials = 10000
version = "scaled_d_hat_test_2" #name of experiment for saving 
plot_dists_true = FALSE

scale_true = TRUE # scale the d_hat value in an effort to normalise the gamma distributions 

upper_bound_X_options = c(50,500,1000)
upper_bound_Y_options = c(50,500,1000)

# Perhaps can think of more creative categorisations
lower_bound_X = 10
lower_bound_Y = 10 

#upper_bound_X = 20
#upper_bound_Y = 20

if (is_local){
  simulation_folder = "/Users/liam/Documents/Resilient Youth/Gemfinder/d_hat_simulation"
  P_matrix_directory = "/Users/liam/Documents/Resilient Youth/Gemfinder/gemfinder/RY_P_matrices/P.RData"
  num_cores = 4
}else{
  simulation_folder = "/home/ubuntu/gemfinder-storage/d_hat_simulation"
  P_matrix_directory = "/home/ubuntu/gemfinder-storage/P.RData"
  
  
  # PACKAGES 
  packages_install = c('ggplot2', 'gridExtra', 'moments', 'data.table', 'fitdistrplus', 
                       'mvtnorm', 'corrplot', 'grid', 'reshape2', 'gamlss', 'doParallel', 'foreach', 'doSNOW', 'tcltk', 'matlib')
  install.packages(packages_install)
  # Install devtools for ggpubr
  if(!require(devtools)) install.packages("devtools")
  devtools::install_github("kassambara/ggpubr")
  
  num_cores = parallel::detectCores()
}

library(ggplot2)
library(mvtnorm)
library(corrplot)
library(gridExtra)
library(moments)
library(ggpubr)
library(reshape2)
library(data.table)
library(grid)
library(gamlss)
library(fitdistrplus)
require(foreach)
require(doParallel)
library(matlib)


package_list = c('ggplot2', 'gridExtra', 'moments', 'data.table', 'fitdistrplus', 
                 'ggpubr', 'mvtnorm', 'corrplot', 'grid', 'reshape2', 'gamlss', 'matlib')
#'doParallel', 'foreach')


# DEFINITIONS ------------------------------------------------------------------

P_matrices = readRDS(P_matrix_directory)

#remove questions with 6 responses - MANUAL for now
drop_questions = c('chs1', 'chs2', 'chs3', 'chs4', 'chs5', 'chs6')
P_matrices = P_matrices[!(names(P_matrices) %in% drop_questions)]

# This is redundant
P_matrix_old = list(
  c(0.8, 0.2, 0,   0  ),
  c(0.2, 0.6, 0.2, 0  ),
  c(0,   0.2, 0.6, 0.2),
  c(0,   0,   0.2, 0.8)
)

num_types = 4 # aka "k" 

prob_vector = function(k) {
  # generate a random probability vector on 1:k.
  
  probs = runif(k)
  return(probs/sum(probs))
}

rdu = function(n, k, distribution) {
  # generate a sample of n students according to some prob distribution on 1:k
  
  sample(1:k, n, replace = T, prob = distribution)
}

random_mix_of_students = function(sample_size, k, P_matrix, given_prob_vec = FALSE) {
  # Generate samples of types from classroom of size sample_size. 
  # k = number of response types
  # types = list of numeric types of sample e.g. c(1,3,3,4,5)
  # P_matrix = dataframe of response probs for each student. P_matrix[i,j] is the probability that student i selects response j. 
  # distribution = randomly generated distribution of types. 
  
  if (isFALSE(given_prob_vec)){
    distribution = prob_vector(k)
  }else{
    distribution = given_prob_vec
  }
  
  random_type = rdu(1, num_types, distribution)
  type_list = c(random_type)
  probs = P_matrix[[ random_type ]]
  
  for (i in 2:sample_size) {
    random_type = rdu(1, num_types, distribution)
    type_list = c(type_list, random_type)
    probs = rbind(probs, P_matrix[[ random_type ]])
  }
  rownames(probs) = NULL
  return(list("types" = type_list, "probs" = probs, "dist" = distribution))
}

rgenmultinom = function(P) {
  # generate a sample of elicited responses given the probs df of random_mix_of_students 
  # obs = vector of total responses for each response e.g. c(10,13,18,21)
  
  obs = matrix(rep(0, num_types), num_types)
  for (k in 1:nrow(P)) {
    obs = obs + rmultinom(1, 1, P[k,])
  }
  return(obs)
}

sigma_maker = function(j, P_matrix) {
  # creates the sigma_j covariance matrix in the d_hat formula
  rho_j = P_matrix[[j]] # row j of P_matrix
  sigma_j = -rho_j %*% t(rho_j)
  diag(sigma_j) = rho_j * (1-rho_j)
  
  return(sigma_j)
}


d_hat_calc = function(X,Y,n,m, sigma, scale_true = FALSE) {
  # calculate d_hat given sampled classrooms
  
  # XY_var_vec = t(X/n^2 + Y/m^2)
  # sigma_hat = XY_var_vec[1]*sigma[[1]]
  # for (k in 2:4){ sigma_hat = sigma_hat + XY_var_vec[k]*sigma[[k]] }
  # 
  # sigma_hat_inv = solve(sigma_hat)
  # print(kappa(sigma_hat))
  # print(eigen(sigma_hat)$values)
  # print(sigma_hat_inv)
  # 
  # d_hat = t(X/n - Y/m) %*% sigma_hat_inv %*% (X/n - Y/m)
  # 
  
  # New d_hat calc with orthogonal projection B 
  
  XY_var_vec = t(X/n^2 + Y/m^2) 
  sigma_hat = XY_var_vec[1]*sigma[[1]]
  for (k in 2:4){ sigma_hat = sigma_hat + XY_var_vec[k]*sigma[[k]] }
  inner_B_sigma = B %*% sigma_hat %*% t(B)
  inner_B_sigma_inv = solve(inner_B_sigma)
  
  #print(eigen(inner_B_sigma)$values)
  
  d_hat = t(B%*%(X/n - Y/m)) %*%  inner_B_sigma_inv %*% (B%*%(X/n - Y/m))
  
  if (scale_true){
    d_hat = 0.5 * (1/n + 1/m) * d_hat
  }
  return(d_hat)
}

path_creator = function(version){
  # check if path has been created, create it if not
  gemfinder_path = simulation_folder
  version_path = paste0(gemfinder_path, "/", version)
  
  if(isFALSE(dir.exists(version_path))){
    dir.create(version_path)
    dir.create(paste0(version_path, "/zoom_in"))
    dir.create(paste0(version_path, "/zoom_out"))
  }
}

moment_calculator = function(prob_dist){
  # Since we have a "pure distribution" we take the population moments
  rv_vec = 1:num_types
  mean = as.numeric(rv_vec %*% prob_dist)
  stdev = as.numeric(sqrt((rv_vec-mean)^2 %*% prob_dist))
  skewness = as.numeric(((rv_vec-mean)^3 %*% prob_dist)/stdev^3)
  kurtosis = as.numeric(((rv_vec-mean)^4 %*% prob_dist)/stdev^4)
  
  return(list(mean=mean, stdev=stdev, skewness=skewness, kurtosis=kurtosis))
}

B_transform = function(n_dim = 4) {
  
  # Populate a matrix of ones.
  B = matrix(rep(1, n_dim^2), nrow = n_dim, ncol = n_dim)
  
  # Make the columns linearly independent.
  for (i in 1:(n_dim-1)) {
    B[i, i+1] = 0
  }
  
  # Make the columns orthonormal.
  B = GramSchmidt(B)
  
  # Transpose so that one of the rows is constant, so that
  # one of the resulting coordinates is degenerate.
  B = t(B)
  
  # Remove that degenerate coordinate.
  B = B[2:n_dim,]
  
}

B = B_transform(4) # This is a 3x4 matrix 


## EXPERIMENTS ----------------------------------------------------------------

# Specifications at top of script
path_creator(version)

# run experiments - each experiment is a different distribution of types. Each trial
# simulates a different distribution of responses, and d_hat is then calculated. 


run_exp = function(exp){
  require(ggplot2)
  require(gridExtra)
  require(moments)
  require(data.table)
  require(fitdistrplus)
  require(ggpubr)
  require(foreach)
  require(doParallel)
  
  print(exp)
  exp_code = paste0(version,"_",exp)
  
  # Get P_matrix 
  random_question = sample(names(P_matrices),1)
  P_matrix = P_matrices[[random_question]]
  noise_matrix = matrix( rnorm(16,mean=0,sd=0.005), num_types, num_types) 
  P_matrix = P_matrix + noise_matrix
  
  # Ensure rows sum to 1
  for (k in  1:nrow(P_matrix)){
    row = P_matrix[k,]
    if (sum(row<0)>0){
      P_matrix[k,] = P_matrix[k,] + abs(min(P_matrix[k,])) + abs(rnorm(1,sd=0.0005))
      
    }
    P_matrix[k,] = P_matrix[k,] / sum(P_matrix[k,])
  }
  P_matrix = round(P_matrix, 3)
  P_matrix = list(P_matrix[1,], P_matrix[2,], P_matrix[3,], P_matrix[4,])
  sigma = list(sigma_maker(1, P_matrix), sigma_maker(2, P_matrix), sigma_maker(3, P_matrix), sigma_maker(4, P_matrix))
  
  
  # Set up classrooms 
  # upper_bound is set in initial settings
  #lower_bound = 10
  #n = 1000
  #m = 1000
  upper_bound_X = sample(upper_bound_X_options, 1)
  upper_bound_Y = sample(upper_bound_Y_options, 1)
  
  n = sample(lower_bound_X:upper_bound_X,1)
  m = sample(lower_bound_Y:upper_bound_Y,1)
  classroom_1 = random_mix_of_students(n, num_types, P_matrix)
  classroom_2 = random_mix_of_students(m, num_types, P_matrix)
  
  #print(0.5*(1/n + 1/m))
  
  # Initialise trials
  d_hat_trials = c()
  
  # Run trials 
  for (i in 1:num_trials){
    X = rgenmultinom(classroom_1$probs)
    Y = rgenmultinom(classroom_2$probs)
    print(X)
    d_hat = d_hat_calc(X,Y,n,m, sigma, scale_true)
    d_hat_trials = c(d_hat_trials, d_hat)
  }
  
  #print(d_hat_trials)
  
  # Build table of X and Y dist 
  X_true = table(factor(classroom_1$types, levels=1:num_types))/n
  Y_true = table(factor(classroom_2$types, levels=1:num_types))/m
  overall_table = rbind(round(X_true,4), round(Y_true,4))
  rownames(overall_table) = c('X', 'Y')
  
  # Calculate Euclidean distance
  euclid_dist = round(sqrt(sum((X_true - Y_true)^2)),4)
  euclid_dist_str = sprintf("%.4f",round(sqrt(sum((X_true - Y_true)^2)),4))
  
  # Fit gamma 
  fit_gamma <- fitdist(as.numeric(d_hat_trials), distr = "gamma", method = "mle")
  shape = fit_gamma$estimate[1]
  rate = fit_gamma$estimate[2]
  #output_df = data.frame(exp_code = exp_code, shape = shape, rate = rate)
  
  # Build feature df 
  X_moments = moment_calculator(X_true)
  Y_moments = moment_calculator(Y_true)
  
  # Change P_matrix from list of vectors to matrix for storage
  P_matrix = do.call(rbind, P_matrix)
  # Because the parallel dopar loop only outputs one row (easily)
  feature_output_df = data.frame(exp_code = exp_code,
                                 X_1 = X_true[1],
                                 X_2 = X_true[2],
                                 X_3 = X_true[3],
                                 X_4 = X_true[4],
                                 Y_1 = Y_true[1],
                                 Y_2 = Y_true[2],
                                 Y_3 = Y_true[3],
                                 Y_4 = Y_true[4],
                                 XmY_1 = abs(X_true[1]-Y_true[1]),
                                 XmY_2 = abs(X_true[2]-Y_true[2]),
                                 XmY_3 = abs(X_true[3]-Y_true[3]),
                                 XmY_4 = abs(X_true[4]-Y_true[4]),
                                 mean_X = X_moments$mean,
                                 stdev_X = X_moments$stdev,
                                 skewness_X = X_moments$skewness,
                                 kurtosis_X = X_moments$kurtosis,
                                 mean_Y = Y_moments$mean,
                                 stdev_Y = Y_moments$stdev,
                                 skewness_Y = Y_moments$skewness,
                                 kurtosis_Y = Y_moments$kurtosis,
                                 euclid_dist_XY = euclid_dist,
                                 n_X = n,
                                 n_Y = m,
                                 P_matrix = 0,
                                 P_matrix_question = random_question,
                                 shape = shape,
                                 rate = rate
  )
  feature_output_df$P_matrix[1] = as.character(list(P_matrix))
  
  # Removing symmetric stuff as its useless - check previous versions if I ever want it again
  
  if (plot_dists_true){
    # Plot histogram of d_hat values
    x_dist = seq(min(d_hat_trials),max(d_hat_trials), length=num_trials)
    x_spot = (max(d_hat_trials)-min(d_hat_trials))*0.8+min(d_hat_trials)
    y_values = dgamma(x=x_dist, shape, rate)
    y_spot_top = (max(y_values)-min(y_values))+min(y_values)
    y_spot_bottom = (max(y_values)-min(y_values))*0.95+min(y_values)
    
    d_hat_trials = data.frame("d_hat"=d_hat_trials)
    
    hist_dens_plot = ggplot(d_hat_trials, aes(x=d_hat, after_stat(density))) + geom_histogram(color='black', fill='white', bins=100) +
      geom_density(alpha=.2, fill="#FF6666") + geom_line(aes(x=x_dist, y=dgamma(x=x_dist, shape, rate)), color="blue", size = 1)  
    
    
    # Build dataframe of metadata
    d_hat_trials = as.numeric(d_hat_trials$d_hat)
    metadata = data.frame(#Euclidean_distance =euclid_dist, 
                          Mean_dhat = round(mean(d_hat_trials),4), 
                          Stdev_dhat = round(sd(d_hat_trials),4),
                          # Skewness_dhat = round(skewness(d_hat_trials),3),
                          # Kurtosis_dhat = round(kurtosis(d_hat_trials),3),
                          Shape = round(shape,3),
                          Rate = round(rate, 3),
                          n_X = n,
                          n_Y = m,
                          n_ratio_X_Y = round(n/m, 3),
                          n_ratio_Y_X = round(m/n, 3),
                          normalise_factor = round(0.5*(1/n+1/m), 3))
    metadata$digits = NULL
    
    # Prepare metadata for plotting
    metadata_transp = transpose(metadata)
    rownames(metadata_transp) = colnames(metadata)
    colnames(metadata_transp) = NULL
    metadata_table_plot = ggtexttable(metadata_transp, theme=ttheme("blank", base_size=10))
    
    dists_table = ggtexttable(overall_table, theme = ttheme(base_size=10))
    
    dists_melted = reshape2::melt(overall_table, id.vars=c("id"), varnames=c("Class", "Type"))
    dists_plot = ggplot(dists_melted, aes(x=Type, y=value, fill=Class)) + geom_col(position="identity", alpha=0.7) + 
      scale_fill_viridis_d() + coord_cartesian(ylim = c(0, 0.8)) + theme(legend.position="top")
    
    
    # Build full graphic of all plots
    layout = rbind(c(1,2),
                   c(1,3), 
                   c(1,4))
    title = paste0(random_question, ", Euclidean distance: ", euclid_dist_str)
    p = grid.arrange(hist_dens_plot, metadata_table_plot, dists_plot, dists_table, ncol=2,nrow=3, layout_matrix=layout, widths=c(0.6,0.4),
                     top = textGrob(title,gp=gpar(fontsize=20,font=3)))
    
    # Save zoomed in
    ggsave(paste0(simulation_folder, "/", version, "/zoom_in/d_hat_rand_classrooms_zoom_in_euclid_dist_", euclid_dist_str,  "_", random_question, ".png"), plot = p)
    
    xlim = c(0,100)
    ylim = c(0, 100)
    if (scale_true){
      xlim = c(0, 3)
      ylim = c(0, 30)
    }
    # Save zoomed out (all on same axis scale)
    hist_dens_plot = hist_dens_plot + coord_cartesian(xlim = xlim, ylim = ylim)
    p = grid.arrange(hist_dens_plot, metadata_table_plot, dists_plot, dists_table, ncol=2,nrow=3, layout_matrix=layout, widths=c(0.6,0.4), 
                     top = textGrob(title,gp=gpar(fontsize=20,font=3)))
    ggsave(paste0(simulation_folder, "/", version, "/zoom_out/d_hat_rand_classrooms_zoom_out_euclid_dist_", euclid_dist_str,  "_", random_question, ".png"), plot = p)
    
  }
  
  return(feature_output_df) 
}

# SERIAL ----------------------------------------------------------------------
# feature_output_df_combined = run_exp(1)
# for (exp in 2:num_exps){
#   feature_output_df_combined = rbind(feature_output_df_combined, run_exp(2))
# }


# PARALLEL --------------------------------------------------------------------
if (testing){
  run_exp(1)
}else {
  
  library(doSNOW)
  library(tcltk)
  
  cl = makeSOCKcluster(num_cores, outfile="")
  registerDoSNOW(cl)
  
  pb = txtProgressBar(max=num_exps, style=3)
  progress = function(n) setTxtProgressBar(pb, n)
  opts = list(progress=progress)
  
  # OLD CODE THAT WORKS - SEEING IF PROGRESS BAR WORKS WITH doSNOW
  #cl = makeCluster(num_cores, outfile="")
  #registerDoParallel(cl)
  
  final_feature_output_df = foreach(exp=1:num_exps, .combine=rbind, .packages=package_list, .errorhandling="remove",
                                    .options.snow=opts
  ) %dopar% {
    feature_output_df = run_exp(exp)
    
    feature_output_df 
  }
  
  stopCluster(cl)
  
  # Output CSV with both features and outputs, and symmetrised data entries 
  write.csv(x=final_feature_output_df, file=paste0(simulation_folder, "/", version, "/feature_output_df.csv"), row.names=FALSE)
  
}



