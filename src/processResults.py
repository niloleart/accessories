import csv

res = dict();
firstRow=[];
with open("results/results.csv", 'rb') as f:
    reader = csv.reader(f)
    first = True

    for row in reader:
        if first:
            firstRow=row;
            first = False
            continue

        if res.has_key(row[0]):
            # per1=((float(row[1])+float(row[2]))/2)
            # per2=((float(res[row[0]][1])+float(res[row[0]][2]))/2)
            per1 = float(row[11])
            per2 = float(res[row[0]][11]);
            if per1 <= 1:
                if (per1 > per2):
                    res[row[0]]=row;
                    #print row, per1
        else:
            # per1 = ((float(row[1]) + float(row[2])) / 2)
            per1 = float(row[11])
            if per1 <= 1:
                res[row[0]] = row;
                #print row

with open("results/processed.csv","wb") as f:
    writer = csv.writer(f)
    writer.writerow(firstRow)
    for k in res.keys():
        print res[k][11]
        writer.writerow(res[k]);
#print res.keys()
