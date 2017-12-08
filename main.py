
# coding: utf-8

# # Data
# 
# ### Source: https://www.kaggle.com/currie32/crimes-in-chicago/data
# Crimes in Chicago from 2001 to 2017 (1.85GB)
# 
# Take 4 csvs then combine them into one file
# ensure to strip header 
# 
# check Primary Type and sanitize it
# select distinct and do sort by name
# visualy observe data
# then sanitize
# 

# In[1]:


from pyspark.sql import functions as func

file_path = "hdfs://sandbox.hortonworks.com:8020/bigdata1_final_project/Chicago_Crimes_2001_to_2004.csv"
crime_df = sqlContext.read.load(file_path,
                        format='com.databricks.spark.csv',
                        header='true',
                        inferSchema=True)

print("Columns:")
for f in crime_df.schema.fields:
    print("%s (%s)" % (f.name, f.dataType))
print('+--------------------------------+')

print(crime_df.select("Primary Type").distinct().count())
crime_df.select("Primary Type").distinct().sort("Primary Type").show(200, False)


# In[3]:


# unique location
crime_df.select("Location Description").distinct().sort("Location Description").show(20, False)


# # Crimes Per Year
# Shows number of crimes reported by year. Sorted by years with highest crime.

# In[3]:


crime_df.groupBy("Year").count().sort("Year").result.show()


# # Arrests Per Year
# Shows number of actual arrests made by year. Not every crime reported resulted in an arrest. Sorted by years with highest arrest number.

# In[6]:


crime_df.where(crime_df.Arrest == "True").groupBy("Year").count().sort("Year").show()


# # Crime to Arrest Ratio
# Calculate ratio of actual arrests from reported crime.
# Create two separate data frames containing year and counts, one with all crimes (all rows) and second with rows that have Arrest => true.

# In[6]:


# Collect Year and Count with arrests only
arrests_only_df = crime_df.where(crime_df.Arrest == "True")
arrests_only_df = arrests_only_df.groupBy("Year").agg(
    func.count(arrests_only_df.ID).alias("arrest_count")).alias('arrests_only')  

# Collect Year and Count with all crimes
all_crime_df = crime_df.groupBy("Year").agg(
    func.count(crime_df.ID).alias("crime_count")).alias('all_crime') 

# Join arrests only and all crime stats counts
joined_df = all_crime_df.alias("all_crime").join(
    arrests_only_df, func.col("all_crime.Year").alias('foo') == func.col("arrests_only.Year")).drop(arrests_only_df.Year)
joined_df = all_crime_df.alias("all_crime").join(arrests_only_df, "Year")
print("*** Crime and Arrest Counts per Year ***")
joined_df.show()

# Get ratio
print("*** Arrest Ratio per Year ***")
joined_df.select(joined_df.Year, (
    (joined_df.arrest_count / joined_df.crime_count) * 100).cast("integer").alias("arrest_ration")
).show()


# # Top Crime per Year
# get count of primary type per year
# then select top one
# 
# figure this out

# In[5]:


crime_df.groupBy("Year", "Primary Type").count().select("count").show()
#crime_df.groupBy("Year", "Primary Type").count().agg("count").show()


# # Top 10 Crimes
# Top 10 crimes, all years.
# Collect count of each crime type per year, then take an average of counts to get top 10 crimes.

# In[4]:


temp_df = crime_df.groupBy("Year", "Primary Type").count().groupBy("Primary Type").agg(
    func.avg("count").cast("integer").alias("Avg Count"))
temp_df.sort("Avg Count", ascending=False).limit(10).show()
#filter("count > 10000").sort("year", "count", ascending=False).show(200, False)
# crime_df.groupBy("Primary Type").count().sort("count", ascending=False).limit(10).show()


# # Top 10 Crimes with Arrests
# Top 10 crimes, all years that resulted in arrests.
# Collect count of each crime type per year that had Arrests=True, then take an average of counts to get top 10 crimes.

# In[5]:


temp_df = crime_df.where(crime_df.Arrest == "True").groupBy("Year", "Primary Type").count().groupBy("Primary Type").agg(
    func.avg("count").cast("integer").alias("Avg Count"))
temp_df.sort("Avg Count", ascending=False).limit(10).show()


# ## Most Dangerous / Safest Hour
# 
# Calculates counts of crimes per date, then take an average of those counts to determine safest / most dangerous hour.
# 

# In[58]:


# TODO: change to do average against counts

crime_df.select(crime_df.Date, crime_df.Year).show()
date_format = "MM/dd/yyyy hh:mm:ss a"

# number 1 approach
crime_df.select(func.date_format(func.unix_timestamp(crime_df.Date, date_format
    ).cast("timestamp"), "MM/dd/yyyy").alias("JustDate"), 
                func.date_format(func.unix_timestamp(crime_df.Date, date_format
    ).cast("timestamp"), "HH").alias("JustHour")
    ).groupBy("JustDate", "JustHour").count().groupBy("JustHour").agg(
        func.avg("count").cast("integer").alias("Avg Count")).sort("Avg Count", ascending=False).show(24, False)
# number 2 approach
hour_counts_df = crime_df.select(func.date_format(func.unix_timestamp(crime_df.Date, date_format
    ).cast("timestamp"), "HH").alias("Hour")).groupBy("Hour").count()

print("*** Order by Crime Count ***")
print("(stay away between 6PM and 2AM, come out between 5AM and 7AM)")

hour_counts_df.sort("count", ascending=False).show(24, False)

# print("*** Ordered by Hour ***")
# hour_counts_df.sort("Hour").show(24, False)


# would also be interesting to add hour per crime type 
# so murders happen at this time the most... robberies at this time the most... etc.
# collect highest crime hour per primary type

# ## Group by location

# ## Use IL Placenames?

# ## Map crimes by districts in Chicago?
