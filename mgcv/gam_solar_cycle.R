# Fitting GAMs to forecast (1) future Solar Cycle Max Amplitude and (2) time of
# future Solar Cycle Max Amplitude
library(mgcv)
library(gridExtra)
library(ggplot2)
library(faraway)
setwd("~/Projects/solarCyclePrediction/mgcv")

################################################################################
GAM_data_amplitude = read.csv("full_data_amplitude.csv", sep=" ")
df_amplitude = data.frame(GAM_data_amplitude)

# (1a): GAM Regression w/ Summary
bpA = gam(FutureMaxAmplitude ~s(CycleEFoldingTime) +
            s(MinAmplitude), data=df_amplitude)
summary(bpA)

# (1b): SC24 Hindcast (save results)
GAM_data_hindcast_sc24_amplitude = read.csv("drivers_for_hindcasting_amplitude.csv", sep=" ")
df_hindcast_sc24_amplitude = data.frame(GAM_data_hindcast_sc24_amplitude)
p_gam_24 = predict(bpA, df_hindcast_sc24_amplitude, se.fit = TRUE)
upr_24 <- p_gam_24$fit + (2 * p_gam_24$se.fit)
lwr_24 <- p_gam_24$fit - (2 * p_gam_24$se.fit)
write.csv(p_gam_24$fit,"gam_hindcasts_amplitude.csv")
write.csv(upr_24,"gam_upr_hindcasts_amplitude.csv")
write.csv(lwr_24,"gam_lwr_hindcasts_amplitude.csv")

# (1c): Prediction of SC25 (save results)
GAM_data_predict_sc25_amplitude = read.csv("drivers_for_forecasting_amplitude.csv", sep=" ")
df_predict_sc25_amplitude = data.frame(GAM_data_predict_sc25_amplitude)
p_gam = predict(bpA, df_predict_sc25_amplitude, se.fit = TRUE)
upr <- p_gam$fit + (2 * p_gam$se.fit)
lwr <- p_gam$fit - (2 * p_gam$se.fit)
write.csv(p_gam$fit,"gam_predictions_amplitude.csv")
write.csv(upr,"gam_upr_amplitude.csv")
write.csv(lwr,"gam_lwr_amplitude.csv")

# (1d): Partial Dependence Plots
dev.new()
plot(bpA, pages=1, all.terms = TRUE, rug = TRUE)
dev.copy2pdf(file='GAM_functions_amplitude.pdf')

################################################################################

# Do the same as the above, but for the time to max amplitude.
GAM_data_amplitude_time = read.csv("full_data_amplitude_time.csv", sep=" ")
df_amplitude_time = data.frame(GAM_data_amplitude_time)

# (2a): GAM Regression w/ Summary
# gam.control(maxit=500)
bpT = gam(FutureMaxAmplitudeTime ~  s(CycleLength), data=df_amplitude_time) #, optimizer="efs")
summary(bpT)

# (2b): SC24 Hindcast (save results)
GAM_data_hindcast_sc24_amplitude_time = read.csv("drivers_for_hindcasting_time.csv", sep=" ")
df_hindcast_sc24_amplitude_time = data.frame(GAM_data_hindcast_sc24_amplitude_time)
p_gam_24_t = predict(bpT, df_hindcast_sc24_amplitude_time, se.fit = TRUE)
upr_24_t <- p_gam_24_t$fit + (2 * p_gam_24_t$se.fit)
lwr_24_t <- p_gam_24_t$fit - (2 * p_gam_24_t$se.fit)
write.csv(p_gam_24_t$fit,"gam_hindcasts_amplitude_time.csv")
write.csv(upr_24_t,"gam_upr_hindcasts_amplitude_time.csv")
write.csv(lwr_24_t,"gam_lwr_hindcasts_amplitude_time.csv")

# (2c): Prediction of SC25 (save results)
GAM_data_predict_sc25_amplitude_time = read.csv("drivers_for_forecasting_time.csv", sep=" ")
df_predict_sc25_amplitude_time = data.frame(GAM_data_predict_sc25_amplitude_time)
p_gam_t = predict(bpT, df_predict_sc25_amplitude_time, se.fit = TRUE)
upr_t <- p_gam_t$fit + (2 * p_gam_t$se.fit)
lwr_t <- p_gam_t$fit - (2 * p_gam_t$se.fit)
write.csv(p_gam_t$fit,"gam_predictions_amplitude_time.csv")
write.csv(upr_t,"gam_upr_amplitude_time.csv")
write.csv(lwr_t,"gam_lwr_amplitude_time.csv")

# (2d): Partial Dependence Plots
dev.new()
plot(bpT, pages=1, all.terms = TRUE, rug = TRUE)
dev.copy2pdf(file='GAM_functions_amplitude_time.pdf')

