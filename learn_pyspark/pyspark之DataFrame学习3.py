import pyspark
spark= pyspark.sql.SparkSession.builder.appName("dataFrame").getOrCreate()
# # 导入类型
from pyspark.sql.types import *
#生成以逗号分隔的数据
stringCSVRDD = spark.sparkContext.parallelize([
    (123,"Katie",19,"brown"),
    (234,"Michael",22,"green"),
    (345,"Simone",23,"blue")
])
#指定模式,StructField(name,dataType,nullable)其中name:该字段的名字，dataType：该字段的数据类型，nullable:指示该字段的值是否为空
schema = StructType([
    StructField("id",LongType(),True),
    StructField("name",StringType(),True),
    StructField("age",LongType(),True),
    StructField("eyeColor",StringType(),True)
])
#对RDD应用该模式并且创建DataFrame
swimmers = spark.createDataFrame(stringCSVRDD,schema)
#利用DataFrame创建一个临时视图
swimmers.registerTempTable("swimmers")
#查看DataFrame的行数
print(swimmers.count())
#获取age=22的id
swimmers.select("id","age").filter("age=22").show()
swimmers.select(swimmers.id,swimmers.age).filter(swimmers.age==22).show()
#获得eyeColor like 'b%'的（name）名字，（eyeColor）眼睛颜色
swimmers.select("name","eyeColor").filter("eyeColor like 'b%'").show()
# swimmers.select("name","eyeColor").filter("eyeColor like 'b%'").show()
spark.sql("select count(1) from swimmers").show()
#用SQL获得age=22的id，age
spark.sql("select id,age from swimmers where age=22").show()
spark.sql("select name,eyeColor from swimmers where eyeColor like 'b%'").show()