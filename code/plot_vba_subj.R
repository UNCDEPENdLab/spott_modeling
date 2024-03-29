#plot 1 subj
#library(R.matlab)
#mfile <- readMat("/Users/mnh5174/Data_Analysis/spott_modeling/outputs/vba_out/ffx/suuvid_base/fit_102_suuvid_base_multisession0.mat")
library(cowplot)
library(tidyverse)


#fix curkey bleed
# s102_raw <- read.csv("/Users/mnh5174/Data_Analysis/spott_modeling/data/vba_input/102_spott_50.csv")
# table(s102_raw$key)
# s102_raw  <- s102_raw %>% arrange(instrial, latency_bin) %>% group_by(instrial) %>%
#   mutate(bin_num=1:n(), first_press=which(key > 0)[1], curkey_mod=if_else(bin_num < first_press, 0L, curkey))
# 
# table(s102_raw$curkey_mod)

setwd("/Users/mnh5174/Data_Analysis/spott_modeling/figures")
trial_stats <- read.csv("/Users/mnh5174/Data_Analysis/spott_modeling/outputs/vba_out/ffx/suuvid_base/pandaa_suuvid_base_ffx_trial_outputs.csv")

#single subj for 102 with correction on clairvoyance
# s102 <- read.csv("/Users/mnh5174/Data_Analysis/spott_modeling/outputs/vba_out/ffx/suuvid_base/pandaa_suuvid_base_ffx_prompt_outputs_onesub.csv") %>% slice(1:1534) %>% filter(id==102) %>% mutate(sample=1:n(), time=sample*50/1000) %>%
#   filter(sample > 100 & sample < 350)

s102 <- trial_stats %>% filter(id==102) %>% mutate(sample=1:n(), time=sample*50/1000) %>%
     filter(sample > 100 & sample < 350)
  
q_df <- s102 %>% gather(key="Action_Value", value="Q", Q1, Q2)
ggplot(q_df, aes(x=time, y=Q, color=Action_Value)) + geom_line()

ggplot(s102, aes(x=time, y=Q1-Q2)) + geom_line() + geom_hline(yintercept=0)

q_df <- s102 %>% mutate(q1_norm=0.3*Q1/max(Q1) + 1.6, q2_norm=0.3*Q2/max(Q2) + 3.2, qdiff=0.6*((Q1-Q2)/max(Q1-Q2)) + 4.6) %>%
  gather(key="Action_Value", value="Q", q1_norm, q2_norm, qdiff)

(qplot <- ggplot(q_df, aes(x=time, y=Q, color=Action_Value)) + geom_line())

choice_df <- s102 %>% gather(key="option", value="chosen", y1, y2, none) %>% 
  gather(key="option_pred", value="chosen_pred", y1_pred, y2_pred, none_pred) %>%
  filter(chosen == 1 & sample > 100 & sample < 350) %>%
  mutate(option_num=dplyr::recode(option, none=0, y1=1, y2=2.6), time=sample*50/1000,
         option_pred_num=dplyr::recode(option_pred, none_pred=0.3, y1_pred=1.3, y2_pred=2.9))
# (cplot <- ggplot(choice_df, aes(x=sample, y=option)) + geom_point(alpha=0.8, size=0.3))

cplot <- ggplot(choice_df, aes(x=time, y=option_num, ymin=option_num-0.1, ymax=option_num+0.1)) + geom_linerange(size=0.5) +
  #geom_tile(aes(x=time, y=option_pred_num, alpha=chosen_pred), color="white", fill="blue", height=0.2) +
  geom_tile(aes(x=time, y=option_pred_num, fill=chosen_pred, color=chosen_pred), height=0.2) +
  geom_line(data=q_df, aes(x=time, y=Q, group=Action_Value, alpha=NULL, ymin=NULL, ymax=NULL), show.legend = FALSE) +
  #scale_alpha_continuous("Model-\npredicted\nprobability", range=c(0.1,.95), breaks=c(0, 0.25, 0.5, 0.75, 1))
  #geom_tile(aes(x=time, y=option_pred_num, color=chosen_pred, fill=chosen_pred), height=0.2, interpolate=TRUE) + 
  #scale_color_viridis_c() + scale_fill_viridis_c()
  scale_fill_distiller("Model-\npredicted\nprobability", palette = "Blues") + scale_color_distiller("Model-\npredicted\nprobability", palette="Blues") +
  geom_hline(yintercept=4.6, linetype="dashed")



#plot_grid(qplot, cplot, ncol=1, align="h")

pdf("s102_timeplot.pdf", width=12, height=6)
cplot + xlab("Time (seconds)") + theme(axis.text.y = element_blank(), axis.title.y = element_blank(), 
                                             axis.ticks.y = element_blank(), axis.line.y = element_blank())
dev.off()

