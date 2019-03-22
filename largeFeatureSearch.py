import math

def main():
	fileName = 'CS205_LARGEtestdata__51.txt'
#	fileName = 'testlarge62.txt'
	f = open(fileName)
	
	dataPoints = []

	for line in f:
		features = []
		fields = line.split()
		classification = int(float(fields[0]))
		for i in range(1,len(fields)):
			features.append(float(fields[i]))
			
		dataPoints.append((classification,features))

	print('Welcome to Robert Colvin\'s Feature Selection Algorithm')
	print('Testing on file:', fileName)
	print('Type the number of the algorithm you want to run')
	print('\t1)Forward Selection')
	print('\t2)Backward Elimination')
	print('\t3)The Special Algorithm')

	algChoice = input()
	bestFeatures = []
	bestAccuracy = 0.0
	if algChoice == '1':
		(bestFeatures,bestAccuracy) = featureSearchForward(dataPoints,100)
	if algChoice == '2':
		(bestFeatures,bestAccuracy) = featureSearchBackward(dataPoints,100)
	if algChoice == '3':
		(bestFeatures,bestAccuracy) = featureSearchForwardPruned(dataPoints,100)

	print('Finished search!! The best feature subset is',bestFeatures,'which has an accuracy of',bestAccuracy*100)

	return

def featureSearchForwardPruned(dataPoints,numFeatures):
	currentFeatures = []
	bestFeatures = []
	bestAccuracy = 0.0

	print('\nThis dataset has',numFeatures,'features (not including the class attribute), with',len(dataPoints),'instances')
	currentFeatures.extend(range(numFeatures-1))
	fullFeatureAccuracy = leaveOneOutCrossValidation(dataPoints,currentFeatures,numFeatures-1,'forward')
	print('\nRunning nearest neighbor with all', numFeatures,'features, using "leaving one out" evaluation, I get an accuracy of',fullFeatureAccuracy*100)
	currentFeatures = []	
	maxIncorrect = len(dataPoints)	
	print('\nBeginning search\n')
	#this loop traverses down the search tree
	for i in range(numFeatures):
		featureToAddAtThisLevel = []
		bestSoFarAccuracy = 0.0
		#this loop traverses over the features
		for k in range(numFeatures):
			if k not in currentFeatures:
				accuracy, maxIncorrect = leaveOneOutCrossValidationPruned(dataPoints,currentFeatures,k,'forward',maxIncorrect)
#				print(maxIncorrect)
				print('\tUsing feature(s) {',k,',',str(currentFeatures).strip('[]'),'}, accuracy is',accuracy*100,'%')
				if accuracy > bestSoFarAccuracy:
					bestSoFarAccuracy = accuracy
					featureToAddAtThisLevel = [k]

		currentFeatures.extend(featureToAddAtThisLevel)
		print()
		if bestSoFarAccuracy > bestAccuracy:
			bestFeatures = currentFeatures.copy()
			bestAccuracy = bestSoFarAccuracy
		elif bestSoFarAccuracy == 0.0:
			print('No additional features improve accuracy. Exiting search')
			break
		elif bestSoFarAccuracy < bestAccuracy:
			print('Warning, Accuracy has decreased! Continuing search in case of local maxima')
		print('Feature set',currentFeatures,'was best, accuracy is',bestSoFarAccuracy*100,'\n')
			
	
	return (bestFeatures,bestAccuracy)

def leaveOneOutCrossValidationPruned(dataPoints,currentFeatures,featureIndex,mode,maxIncorrect):
#	print(currentFeatures,featureIndex)
	numCorrectClassifications = 0.0
	numIncorrect = 0.0
	numAttempts = 0.0
	for leftOutPoint in dataPoints:
		nearestNeighbor = 0
		nearestDistance = 100000000000000000000000000000.0
		for compareAgainstPoint in dataPoints:
			if leftOutPoint != compareAgainstPoint:
				features1 = []
				features2 = []
				for cfIndex in currentFeatures:
					if cfIndex != featureIndex:
						features1.append(leftOutPoint[1][cfIndex])
						features2.append(compareAgainstPoint[1][cfIndex])
				if mode == 'forward':
					features1.append(leftOutPoint[1][featureIndex])
					features2.append(compareAgainstPoint[1][featureIndex])
		#		print(features1,features2)
				
				distance = euclideanDistance(features1,features2)
				if distance < nearestDistance:
					nearestDistance = distance
					nearestNeighbor = compareAgainstPoint[0]
		if nearestNeighbor == leftOutPoint[0]:
			numCorrectClassifications += 1.0
		else:
			numIncorrect += 1
			if numIncorrect > maxIncorrect:
				return (0.0, maxIncorrect)
		numAttempts += 1.0

#	print(featureIndex,'got correct',numCorrectClassifications,'outta',numAttempts)
	return (numCorrectClassifications/numAttempts, numIncorrect)
def featureSearchBackward(dataPoints,numFeatures):
	currentFeatures = []
	bestFeatures = []
	bestAccuracy = 0.0

	print('\nThis dataset has',numFeatures,'features (not including the class attribute), with',len(dataPoints),'instances')
	currentFeatures.extend(range(numFeatures))
	fullFeatureAccuracy = leaveOneOutCrossValidation(dataPoints,currentFeatures,numFeatures-1,'backward')
	print('\nRunning nearest neighbor with all', numFeatures,'features, using "leaving one out" evaluation, I get an accuracy of',fullFeatureAccuracy*100)
		
	print('\nBeginning search\n')
	#this loop traverses down the search tree
	for i in range(numFeatures):
		featureToKillAtThisLevel = -1
		bestSoFarAccuracy = 0.0
		#this loop traverses over the features
		for k in range(numFeatures):
			if k in currentFeatures:
				accuracy = leaveOneOutCrossValidation(dataPoints,currentFeatures,k,'backward')
				printFeatures = currentFeatures.copy()
				printFeatures.remove(k)
				print('\tUsing feature(s) {',str(printFeatures).strip('[]'),'}, accuracy is',accuracy*100,'%')
				if accuracy > bestSoFarAccuracy:
					bestSoFarAccuracy = accuracy
					featureToKillAtThisLevel = k

		currentFeatures.remove(featureToKillAtThisLevel)
#		currentFeatures = [x for x in currentFeatures if x not in featureToKillAtThisLevel]
		print()
		if bestSoFarAccuracy > bestAccuracy:
			bestFeatures = currentFeatures.copy()
			bestAccuracy = bestSoFarAccuracy
		elif bestSoFarAccuracy < bestAccuracy:
			print('Warning, Accuracy has decreased! Continuing search in case of local maxima')
		print('Feature set',currentFeatures,'was best, accuracy is',bestSoFarAccuracy*100,'\n')
			
	
	return (bestFeatures,bestAccuracy)

def featureSearchForward(dataPoints,numFeatures):
	currentFeatures = []
	bestFeatures = []
	bestAccuracy = 0.0

	print('\nThis dataset has',numFeatures,'features (not including the class attribute), with',len(dataPoints),'instances')
	currentFeatures.extend(range(numFeatures-1))
	fullFeatureAccuracy = leaveOneOutCrossValidation(dataPoints,currentFeatures,numFeatures-1,'forward')
	print('\nRunning nearest neighbor with all', numFeatures,'features, using "leaving one out" evaluation, I get an accuracy of',fullFeatureAccuracy*100)
	currentFeatures = []	
	
	print('\nBeginning search\n')
	#this loop traverses down the search tree
	for i in range(numFeatures):
		featureToAddAtThisLevel = []
		bestSoFarAccuracy = 0.0
		#this loop traverses over the features
		for k in range(numFeatures):
			if k not in currentFeatures:
				accuracy = leaveOneOutCrossValidation(dataPoints,currentFeatures,k,'forward')
				print('\tUsing feature(s) {',k,',',str(currentFeatures).strip('[]'),'}, accuracy is',accuracy*100,'%')
				if accuracy > bestSoFarAccuracy:
					bestSoFarAccuracy = accuracy
					featureToAddAtThisLevel = [k]

		currentFeatures.extend(featureToAddAtThisLevel)
		print()
		if bestSoFarAccuracy > bestAccuracy:
			bestFeatures = currentFeatures.copy()
			bestAccuracy = bestSoFarAccuracy
		elif bestSoFarAccuracy < bestAccuracy:
			print('Warning, Accuracy has decreased! Continuing search in case of local maxima')
		print('Feature set',currentFeatures,'was best, accuracy is',bestSoFarAccuracy*100,'\n')
			
	
	return (bestFeatures,bestAccuracy)

def euclideanDistance(features1, features2):
	distance = 0
	for i in range(len(features1)):
		distance += math.pow((features1[i] - features2[i]),2)
	distance = math.sqrt(distance)
	return distance

def leaveOneOutCrossValidation(dataPoints,currentFeatures,featureIndex,mode):
#	print(currentFeatures,featureIndex)
	numCorrectClassifications = 0.0
	numAttempts = 0.0
	for leftOutPoint in dataPoints:
		nearestNeighbor = 0
		nearestDistance = 100000000000000000000000000000.0
		for compareAgainstPoint in dataPoints:
			if leftOutPoint != compareAgainstPoint:
				features1 = []
				features2 = []
				for cfIndex in currentFeatures:
					if cfIndex != featureIndex:
						features1.append(leftOutPoint[1][cfIndex])
						features2.append(compareAgainstPoint[1][cfIndex])
				if mode == 'forward':
					features1.append(leftOutPoint[1][featureIndex])
					features2.append(compareAgainstPoint[1][featureIndex])
		#		print(features1,features2)
				
				distance = euclideanDistance(features1,features2)
				if distance < nearestDistance:
					nearestDistance = distance
					nearestNeighbor = compareAgainstPoint[0]
		if nearestNeighbor == leftOutPoint[0]:
			numCorrectClassifications += 1.0
		numAttempts += 1.0

#	print(featureIndex,'got correct',numCorrectClassifications,'outta',numAttempts)
	return (numCorrectClassifications/numAttempts)


if __name__ == '__main__':
	main()
