# created in July 07, 2025
# refer to guorong dai jrssb 2024 and haoze hou jbes 2025
# Download data from: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015

library(haven)
library(dplyr)

# 1. Read XPT files
## demographics data- Demographic variables and sample weights
df = read_xpt("data/NHANES/2015/DEMO_I.XPT")   
# RIAGENDR: gender (1 for male, 2 for female, 0 for missing); RIDAGEYR: Age in years at screening; DMDHRAGE: Household Reference Person's Age in Years; 
# RIDRETH3: Race/Hispanic origin w/ NH Asian: 1	Mexican American; 2	Other Hispanic; 3	Non-Hispanic White; 4	Non-Hispanic Black; 6	Non-Hispanic Asian; 7	Other Race-Including Multi-Racial
# DMDEDUC2: education for adults 20+, 1	Less than 9th grade; 2	9-11th grade; 3	High school graduate/GED; 4	Some college or AA degree; 5 College graduate

## examination data - Body Measures
df2 = read_xpt("data/NHANES/2015/BMX_I.XPT")    
# head(df2)
# BMXBMI: BMI (kg/m^2), BMIWT: weight (kg)

## examination data - Blood Pressure
df3 = read_xpt("data/NHANES/2015/BPX_I.XPT")    
head(df3)

# BPXSY1, BPXSY2, BPXSY3, BPXSY4: systolic blood pressure  ????????? (mmHg) 
# BPXDI1, BPXDI2, BPXDI3, BPXDI4: diastolic blood pressure ????????? (mmHg)
# BPXPULS: Pulse regular or irregular? ??????????????????

## dietary data - total nutrient intakes, two 24-hour dietary recall interviews
df4 = read_xpt("data/NHANES/2015/DR1TOT_I.XPT")  # first day
df5 = read_xpt("data/NHANES/2015/DR2TOT_I.XPT")  # second day
head(df4)
head(df5)
# DR1TKCAL: Energy (kcal); DR1TPROT: Protein (gm); DR1TCARB: Carbohydrate (gm); DR1TSUGR: Total sugars (gm) ???
# DR1TTFAT: Total fat (gm); DR1TCHOL: Cholesterol (mg); DR1TALCO: Alcohol (gm); DR1TSODI: Sodium (mg) ??????
# DR1TSFAT: Total saturated fatty acids (gm)

## questionnaire data - Smoking, Cigarette Use; also include diabetes data
df6 = read_xpt("data/NHANES/2015/SMQ_I.xpt") # SMQ040: Do you now smoke cigarettes? 1	Every day; 2	Some days; 3	Not at all
df7 = read_xpt("data/NHANES/2015/PAQ_I.xpt") # PAQ655: Days vigorous recreational activities. 1-7 days; 99 don't know; 77 refused
df8 = read_xpt("data/NHANES/2015/SLQ_I.xpt") # SLD012: Sleep hours

# 2. Merge dataframes
df <- merge(df, df2, by = "SEQN", all.x = TRUE)
df <- merge(df, df3, by = "SEQN", all.x = TRUE)
df <- merge(df, df4, by = "SEQN", all.x = TRUE)
df <- merge(df, df5, by = "SEQN", all.x = TRUE)
df <- merge(df, df6, by = "SEQN", all.x = TRUE)
df <- merge(df, df7, by = "SEQN", all.x = TRUE)
df <- merge(df, df8, by = "SEQN", all.x = TRUE)
head(df)

# 3. Select columns and rename
final <- df[, c("RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "BMXBMI", "BMIWT", "SMQ040", "PAQ655", "SLD012",
                "BPXPULS", "BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2", "BPXSY3", "BPXDI3", "BPXSY4", "BPXDI4",
                "DR1TTFAT", "DR1TSFAT", "DR1TKCAL", "DR1TSUGR", "DR1TPROT", "DR1TCARB", "DR1TSODI", "DR1TCHOL", "DR1TALCO",
                "DR2TTFAT", "DR2TSFAT", "DR2TKCAL", "DR2TSUGR", "DR2TPROT", "DR2TCARB", "DR2TSODI", "DR2TCHOL", "DR2TALCO")]
colnames(final) <- c("sex", "age", "race", "education", "bmi", "weight", "smoke", "sport","sleep",
                     "pulse", "SY1", "DI1", "SY2", "DI2", "SY3", "DI3", "SY4", "DI4",
                     "fat1", "sfat1", "energy1", "sugar1", "protein1", "carbohydrate1", "sodium1", "cholesterol1", "alcohol1",
                     "fat2", "sfat2", "energy2", "sugar2", "protein2", "carbohydrate2", "sodium2", "cholesterol2", "alcohol2")
## SY: systolic blood pressure; DI: diastolic blood pressure # sample size = 9971


# 4. Obtain final data
# Filter out NAs

final_without_na <- final[!is.na(final$SY1) & !is.na(final$DI1) & 
                            !is.na(final$SY2) & !is.na(final$DI2) & 
                            !is.na(final$SY3) & !is.na(final$DI3) & 
                            is.na(final$SY4) & is.na(final$DI4), ]

final_without_na <- final_without_na[!is.na(final_without_na$fat1) & 
                                       !is.na(final_without_na$fat2), ]

# Filter rows where DI1 equals 0
head(final_without_na[final_without_na$DI1 == 0, ])

# Summary statistics
summary(final_without_na)

# Calculate MeanSY (the outcome was measured multiple times to alleviate measurement errors)
final_without_na$MeanSY <- (final_without_na$SY1 + final_without_na$SY2 + final_without_na$SY3) / 3

