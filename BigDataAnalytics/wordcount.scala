val data = sc.textFile("/home/pict/Desktop/spark.txt")
val splitData = data.flatMap(line => line.split(" "))
val mapData = splitData.map(word => (word, 1))
val reduceData = mapData.reduceByKey((_+_))
reduceData.collect
