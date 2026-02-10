library(tidyverse)
library(latex2exp)
library(gridExtra)
library(grid)
library(here)

### Figure 1

df_sc_B = read.csv(here("output", "changeB_seed.csv"), header = TRUE, col.names = c("B", "R", "c", "method", "R.squared"))
df_sc_R = read.csv(here("output", "changeR_seed.csv"), header = TRUE, col.names = c("B", "R", "c", "method", "R.squared"))
df_sc_c = read.csv(here("output", "changeC_seed.csv"), header = TRUE, col.names = c("B", "R", "c", "method", "R.squared"))

df_sc = rbind(df_sc_B, df_sc_R, df_sc_c)

df_sc$method = factor(df_sc$method, 
                       levels = c("ACW", "DR", "SR", "COKE"),
                       labels = c("ACW-CATE", "DR-CATE", "SR", "COKE"))

c_fix = 1
R_fix = 2
B_fix = 10

# p1: changeB (vary S_B with other parameters fixed under q=1)

df_filtered = df_sc %>%
  filter(c == c_fix, R == R_fix)

df_means = df_filtered %>%
  group_by(B, method) %>%
  summarize(mean_r_squared = mean(`R.squared`))

p1 = ggplot(df_means, aes(x = as.factor(B), y = mean_r_squared, color = method, shape = method, group = method)) +
  geom_point(size = 1.5) +
  geom_line() +
  ylab("MSE") +
  xlab(TeX("$S_B$ (Degree of Shift b/w Source & Target)")) +
  geom_hline(yintercept = 0.6, linetype = "dashed", color = "gray") +
  coord_cartesian(ylim = c(0.03, 4.5)) +
  scale_y_continuous(
    breaks = c(0, 0.2, 0.4,  0.6, 5),
    trans = scales::trans_new(
      "compressed_top",
      transform = function(x) ifelse(x > 0.6, (x - 0.6) / 60 + 0.6, x),
      inverse = function(x) ifelse(x > 0.6, (x - 0.6) * 60 + 0.6, x)
    )
  ) +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12)
  )
p1

# p2: changeR (vary S_R with other parameters fixed under q=1)

df_filtered = df_sc %>%
  filter(c == c_fix, B == B_fix) %>%
  filter(R != 5)

df_means = df_filtered %>%
  group_by(R, method) %>%
  summarize(mean_r_squared = mean(`R.squared`))

p2 = ggplot(df_means, aes(x = as.factor(R), y = mean_r_squared, color = method, shape = method, group = method)) +
  geom_point(size = 1.5) +
  geom_line()  +
  geom_hline(yintercept = 0.6, linetype = "dashed", color = "gray") +
  ylab("MSE") +
  xlab(TeX("$S_R$ (Degree of Shift b/w Treated & Control)")) +
  coord_cartesian(ylim = c(0.03, 12)) +
  scale_y_continuous(
    breaks = c(0, 0.2, 0.4,  0.6, 12),
    trans = scales::trans_new(
      "compressed_top",
      transform = function(x) ifelse(x > 0.6, (x - 0.6) / 200 + 0.6, x),
      inverse = function(x) ifelse(x > 0.6, (x - 0.6) * 200 + 0.6, x)
    )
  ) +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12)
  )
p2

# p3: changeC (vary S_C with other parameters fixed under q=1)

df_filtered = df_sc %>%
  filter(R == R_fix, B == B_fix) %>%
  filter(c %in% c(0,0.25,0.5,0.75,1, 1.25))

df_means = df_filtered %>%
  group_by(c, method) %>%
  summarize(mean_r_squared = mean(`R.squared`))

p3 = ggplot(df_means, aes(x = as.factor(c), y = mean_r_squared, color = method, shape = method, group = method)) +
  geom_point(size = 1.5) +
  geom_line() +
  geom_hline(yintercept = 0.6, linetype = "dashed", color = "gray") +
  ylab("MSE") +
  xlab(TeX("c (Complexity of $f_a^*$ compared to $h^*$)")) +
  coord_cartesian(ylim = c(0.03, 4)) +
  scale_y_continuous(
    breaks = c(0, 0.2, 0.4,  0.6, 4),
    trans = scales::trans_new(
      "compressed_top",
      transform = function(x) ifelse(x > 0.6, (x - 0.6) / 50 + 0.6, x),
      inverse = function(x) ifelse(x > 0.6, (x - 0.6) * 50 + 0.6, x)
    )
  ) +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12)
  )
p3

# p4: changeB_2dim (vary S_B with other parameters fixed under q=2)

df_sc = read.csv(here("output", "changeB_2dim.csv"), header = TRUE, col.names = c("B", "R", "c", "method", "R.squared"))

df_sc$method = factor(df_sc$method, 
                       levels = c("ACW", "DR", "SR", "COKE"),
                       labels = c("ACW-CATE", "DR-CATE", "SR", "COKE"))

c_fix = 1
R_fix = 2
B_fix = 10

df_filtered = df_sc %>%
  filter(c == c_fix, R == R_fix)

df_means = df_filtered %>%
  group_by(B, method) %>%
  summarize(mean_r_squared = mean(`R.squared`))

p4 = ggplot(df_means, aes(x = as.factor(B), y = mean_r_squared, color = method, shape = method, group = method)) +
  geom_point(size = 1.5) +
  geom_line() +
  geom_hline(yintercept = 0.3, linetype = "dashed", color = "gray") +
  ylab("MSE") +
  xlab(TeX("$S_B$ (for 2-dimensional CATE)")) +
  coord_cartesian(ylim = c(0.015, 1.2)) +
  scale_y_continuous(
    breaks = c(0, 0.1, 0.2,  0.3, 1.2),
    trans = scales::trans_new(
      "compressed_top",
      transform = function(x) ifelse(x > 0.3, (x - 0.3) / 20 + 0.3, x),
      inverse = function(x) ifelse(x > 0.3, (x - 0.3) * 20 + 0.3, x)
    )
  ) +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12)
  )
p4

# p5: changeN (vary n_T = n/4 with other parameters fixed under q=1)

df_sc = read.csv(here("output", "changeN_seed.csv"), header = TRUE, col.names = c("nt", "method", "R.squared"))

df_sc$method = factor(df_sc$method, 
                       levels = c("ACW", "DR", "SR", "COKE"),
                       labels = c("ACW-CATE", "DR-CATE", "SR", "COKE"))

df_filtered = df_sc

df_means = df_filtered %>%
  group_by(nt, method) %>%
  summarize(mean_r_squared = mean(`R.squared`))

p5 = ggplot(df_means, aes(x = as.factor(nt), y = mean_r_squared, color = method, shape = method, group = method)) +
  geom_point(size = 1.5) +
  geom_line() +
  geom_hline(yintercept = 0.6, linetype = "dashed", color = "gray") +
  ylab("MSE") +
  xlab("n_T (= n_S/4)") +
  coord_cartesian(ylim = c(0.03, 9)) +
  scale_y_continuous(
    breaks = c(0, 0.2, 0.4,  0.6, 9),
    trans = scales::trans_new(
      "compressed_top",
      transform = function(x) ifelse(x > 0.6, (x - 0.6) / 100 + 0.6, x),
      inverse = function(x) ifelse(x > 0.6, (x - 0.6) * 100 + 0.6, x)
    )
  ) +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12)
  )
p5

# Combine p1–p5 into a 2+2+1 layout

grid.arrange(
  p1, p2,
  nullGrob(),
  p3, p4,
  nullGrob(),
  p5,
  nrow = 5,
  heights = c(1, 0.1, 1, 0.1, 1)
)

grid.arrange(
  p1, p2,
  nullGrob(), nullGrob(),
  p3, p4,
  nullGrob(), nullGrob(),
  p5, nullGrob(),
  nrow = 5,
  heights = c(1, 0.15, 1, 0.15, 1)
)



### Figure A1

# changeB_CF (compare the cross-fitting version of COKE with the original Algorithm 3)

df_sc = read.csv(here("output", "changeB_CF.csv"), header = TRUE, col.names = c("B", "CF", "R.squared"))

df_sc$CF = factor(df_sc$CF, 
                   levels = c("perm1","perm2", "all"),
                   labels = c("NO","NO2", "YES"))

df_means = df_sc %>%
  filter(CF != 'NO2') %>%
  group_by(B, CF) %>%
  summarize(mean_r_squared = mean(`R.squared`))

ggplot(df_means, aes(x = as.factor(B), y = mean_r_squared, color = CF, shape = CF, group = CF)) +
  geom_point(size = 1.5) +
  geom_line() +
  ylab("MSE") +
  xlab(TeX("$S_B$ (Degree of Covariate Shift between Source & Target)")) +
  coord_cartesian(ylim = c(0.1,0.25)) +
  theme(
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12)
  )
