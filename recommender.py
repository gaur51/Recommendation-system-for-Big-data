
# coding: utf-8

# #### Preparing the Data

# In[1]:

# load data

from pyspark import SparkContext

sc = SparkContext()
rawUserArtistData = sc.textFile("audio_data/user_artist_data.txt")
rawArtistData = sc.textFile("audio_data/artist_data.txt")
rawArtistAlias = sc.textFile("audio_data/artist_alias.txt")

# show stats
userIDs = rawUserArtistData.map(lambda l: float(l.split(' ')[0]) )
artistIDs = rawUserArtistData.map(lambda l: float(l.split(' ')[1]) )
print userIDs.stats
# In[2]:

def processArtistByID(line):
    l = line.split('\t')
    if (len(l) != 2):
        return (-1,-1)
    else:
        try:
            id = int(l[0])
        except:
            return (-1,-1)
        return (id, l[1])
artistByID = rawArtistData.map(lambda l: processArtistByID(l)).filter(lambda l: l[0] != -1)
print(artistByID.take(5))

def processArtistAlias(line):
    l = line.split('\t')
    if (len(l) != 2):
        return (-1,-1)
    else:
        try:
            id = int(l[0])
        except:
            return (-1,-1)
        return (id, int(l[1]))
artistAlias = rawArtistAlias.map(lambda l: processArtistAlias(l) ).filter(lambda l: l[0] != -1)
bArtistAlias = sc.broadcast(artistAlias.collectAsMap())


# #### Building a First Model

# In[3]:

from pyspark.mllib.recommendation import Rating
def processData(line):
    userID, artistID, count = [int(x) for x in line.split(' ')]
    artistID = bArtistAlias.value.get(artistID) or artistID
    return Rating(userID, artistID, count)
allData = rawUserArtistData.map(lambda l: processData(l)).cache()
allData.take(5)


# In[4]:

from pyspark.mllib.recommendation import ALS
# Build the recommendation model using Alternating Least Squares based on implicit ratings
model = ALS.trainImplicit(ratings = allData, rank = 10, iterations = 5, lambda_=0.01, alpha=1.0)
# http://spark.apache.org/docs/latest/mllib-collaborative-filtering.html
allData.unpersist()


# In[5]:

print model.userFeatures().mapValues(list).first()


# #### Spot Checking Recommendations

# In[6]:

userID = 2093760
rawArtistsForUser = rawUserArtistData.map(lambda l: l.split(' ')).filter(lambda l: l[0] == str(userID)) 
existingProducts = set(rawArtistsForUser.map(lambda l: int(l[1])).collect() )
print artistByID.filter(lambda l: existingProducts.__contains__(l[0]) ).take(5)
recommendations = list(model.call("recommendProducts", userID, 10)) # model.recommendProducts(UserID, 5)
# ref: https://spark.apache.org/docs/1.5.1/api/python/_modules/pyspark/mllib/recommendation.html
recommendedProductIDs = set(l.product for l in recommendations)
for item in artistByID.filter(lambda l: recommendedProductIDs.__contains__(l[0]) ).collect():
    print item


# In[ ]:



