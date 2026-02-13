# created in July 07, 2025
# refer to guorong dai jrssb 2024 and haoze hou jbes 2025
# Download data from: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2001

library(haven)
library(dplyr)

# 1. Read XPT files
df = read_xpt("data/NHANES/2001/DEMO_B.XPT")   
df2 = read_xpt("data/NHANES/2001/BMX_B.XPT")    
df3 = read_xpt("data/NHANES/2001/BPX_B.XPT")    
df4 = read_xpt("data/NHANES/2001/DRXTOT_B.XPT") # only one day's measure
df6 = read_xpt("data/NHANES/2001/SMQ_B.xpt")


# 2. Merge dataframes
df <- merge(df, df2, by = "SEQN", all.x = TRUE)
df <- merge(df, df3, by = "SEQN", all.x = TRUE)
df <- merge(df, df4, by = "SEQN", all.x = TRUE)
df <- merge(df, df6, by = "SEQN", all.x = TRUE)

# 3. Select columns and rename
final <- df[, c("RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "BMXBMI", "BMIWT", "SMQ040",
                "BPXPULS", "BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2", "BPXSY3", "BPXDI3", "BPXSY4", "BPXDI4",
                "DRXTTFAT", "DRXTSFAT", "DRXTKCAL", "DRXTSUGR", "DRXTPROT", "DRXTCARB", "DRDTSODI", "DRXTCHOL", "DRXTALCO")]
colnames(final) <- c("sex", "age", "race", "education", "bmi", "weight", "smoke",
                     "pulse", "SY1", "DI1", "SY2", "DI2", "SY3", "DI3", "SY4", "DI4",
                     "fat1", "sfat1", "energy1", "sugar1", "protein1", "carbohydrate1", "sodium1", "cholesterol1", "alcohol1")
## SY: systolic blood pressure; DI: diastolic blood pressure # sample size = 11039


# 4. Obtain final data
# Filter out NAs
final_without_na <- final[!is.na(final$SY1) & !is.na(final$DI1) & 
                            !is.na(final$SY2) & !is.na(final$DI2) & 
                            !is.na(final$SY3) & !is.na(final$DI3) & 
                            is.na(final$SY4) & is.na(final$DI4), ]

final_without_na <- final_without_na[!is.na(final_without_na$fat1) & 
                                       !is.na(final_without_na$fat2),]

# Calculate MeanSY (the outcome was measured multiple times to alleviate measurement errors)
final_without_na$MeanSY <- (final_without_na$SY1 + final_without_na$SY2 + final_without_na$SY3) / 3
