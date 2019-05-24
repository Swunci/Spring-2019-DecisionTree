import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Random;

public class DecisionTree {

	public static double getInformationGain(ArrayList<Double> output, ArrayList<Double> feature, double entropy) {
		double gain = entropy - getConditionalEntropy(output, feature);
		return gain;
	}
	
	public static double getEntropy(ArrayList<Double> output, ArrayList<Double> feature, double threshold) {
		ArrayList<Double> featureOutputs1 = new ArrayList<>();
		ArrayList<Double> featureOutputs2 = new ArrayList<>();
		for (int i = 0; i < feature.size(); i++) {
			if (feature.get(i) < threshold) {
				featureOutputs1.add(output.get(i));
			}
			else {
				featureOutputs2.add(output.get(i));
			}
		}
		double fraction1 = featureOutputs1.size() / (double) output.size();
		double fraction2 = featureOutputs2.size() / (double) output.size();
		double minEntropy = fraction1 *getEntropy(featureOutputs1) + fraction2 * getEntropy(featureOutputs2);
		return minEntropy;
	}
	
	public static double getThreshold(ArrayList<Double> output, ArrayList<Double> feature) {
		ArrayList<Double> uniqueValues = getUniqueValues(feature);
		// Need to initialize minEntropy
		ArrayList<Double> featureOutputs1 = new ArrayList<>();
		ArrayList<Double> featureOutputs2 = new ArrayList<>();
		double t = feature.get(0);
		for (int i = 0; i < feature.size(); i++) {
			if (feature.get(i) < t) {
				featureOutputs1.add(output.get(i));
			}
			else {
				featureOutputs2.add(output.get(i));
			}
		}
		double fraction1 = featureOutputs1.size() / (double) output.size();
		double fraction2 = featureOutputs2.size() / (double) output.size();
		double minEntropy = fraction1 *getEntropy(featureOutputs1) + fraction2 * getEntropy(featureOutputs2);
		int index = 0;
		int counter = 0;
		// Find the entropy of each threshold and get the minEntropy
		for (double threshold : uniqueValues) {
			featureOutputs1 = new ArrayList<>();
			featureOutputs2 = new ArrayList<>();
			for (int i = 0; i < feature.size(); i++) {
				if (feature.get(i) < threshold) {
					featureOutputs1.add(output.get(i));
				}
				else {
					featureOutputs2.add(output.get(i));
				}
			}
			fraction1 = featureOutputs1.size() / (double) output.size();
			fraction2 = featureOutputs2.size() / (double) output.size();
			double entropy = fraction1 *getEntropy(featureOutputs1) + fraction2 * getEntropy(featureOutputs2);
			if (entropy < minEntropy) {
				minEntropy = entropy;
				index = counter;
			}
			counter++;	
		}
		return uniqueValues.get(index);
	}
	
	public static double getConditionalEntropy(ArrayList<Double> output, ArrayList<Double> feature) {
		ArrayList<Double> uniqueValues = getUniqueValues(feature);
		double conditionalEntropy = 0;
		if (uniqueValues.size() > 2) {
			double threshold = getThreshold(output, feature);
			conditionalEntropy = getEntropy(output, feature, threshold);
		}
		else {
			// Find all the outputs for each category of the feature and calculate the entropy
			for (double value : uniqueValues) {
				ArrayList<Double> featureOutputs = new ArrayList<>();
				for (int i = 0; i < feature.size(); i++) {
					if (value == feature.get(i)) {
						featureOutputs.add(output.get(i));
					}
				}
				// Probability of this category happening times entropy of the outputs of this category
				conditionalEntropy += (featureOutputs.size() / (double) output.size()) * getEntropy(featureOutputs);
			}
		}
		return conditionalEntropy;
	}
	
	public static double getEntropy(ArrayList<Double> output) {
		ArrayList<Double> uniqueValues = getUniqueValues(output);
		double entropy = 0;
		for (double value : uniqueValues) {
			double counter = 0;
			for (double num : output) {
				if (num == value) {
					counter++;
				}
			}
			double fraction = counter/ (double) output.size();
			entropy -= fraction * (Math.log(fraction) / Math.log(2));
		}
		return entropy;
	}

	public static ArrayList<Double> getUniqueValues(ArrayList<Double> feature) {
		ArrayList<Double> uniqueValues = new ArrayList<Double>();
		for (double num : feature) {
			if (!uniqueValues.contains(num)) {
				uniqueValues.add(num);
			}
		}
		return uniqueValues;
	}
	
	public static double calculateAccuracy(Node treeRoot, ArrayList<ArrayList<Double>> data) {
		double correct = 0;
		for (int i = 0; i < data.get(0).size(); i++) {
			Node node = treeRoot;
			while(!node.isLeaf()) {
				int feature = node.getFeature();
				// If the feature was split on a threshold
				if (node.thresholdUsed()) {
					double threshold = node.getThreshold();
					if (data.get(feature).get(i) < threshold) {
						node = node.getChildren().get(0);
					}
					else {
						node = node.getChildren().get(1);
					}
					continue;
				}
				// Compare feature value at i to the value of each children
				else {
					for (int j = 0; j < node.getChildren().size(); j++) {
						if (data.get(feature).get(i) == node.getChildren().get(j).getSplitValue()) {
							node = node.getChildren().get(j);
							break;
						}
						// if feature value at i does not match any of the values of the children, pick a random branch
						if (j == node.getChildren().size() - 1) {
							Random rand = new Random();
							int randNum = rand.nextInt(j+1);
							node = node.getChildren().get(randNum);
						}
					}
				}
			}
			// Finished traversing the tree and reaching a leaf node
			// Compare the leaf node prediction with feature output
			if (data.get(0).get(i) == node.getPrediction()) {
				correct++;
			}
		}
		return correct/data.get(0).size() * 100;
	}
	
	public static Node buildDecisionTree(ArrayList<ArrayList<Double>> data, ArrayList<Integer> featuresUsed, int depth) {
		// If all output have the same value (Base case 1)
		double majority = 0;
		double majorityOutput = 0;
		ArrayList<Double> outputValues = getUniqueValues(data.get(0));
		for (double output : outputValues) {
			int counter = 0;
			for (double value : data.get(0)) {
				if (value == output) {
					counter++;
				}
			}
			if (counter > majority) {
				majority = counter;
				majorityOutput = output;
			}
		}
		// If majority is equal to the number of outputs that means all output have same value
		if (majority == data.get(0).size()) {
			Node leaf = new Node();
			leaf.setPrediction(majorityOutput);
			return leaf;
		}
		// If the all the features have the same input (Base case 2)
		if (featuresUsed.size() != 0) {
			for (int i = 0; i < data.size(); i++) {
				boolean same = true;
				for (int j = 1; i < data.get(i).size(); j++) {
					// If the current input is not equal to the previous one, then not all inputs are of the same value
					if (!(data.get(i).get(j).equals(data.get(i).get(j - 1)))) {
						same = false;
						break;
					}
				}
				if (!same) {
					break;
				}
				// If loops reaches the last input of the last feature
				if (i == data.size() - 1) {
					Node leaf = new Node();
					leaf.setPrediction(majorityOutput);
					return leaf;
				}
			}
			
		}
		// If max depth is reached (Base case 3)
		if (featuresUsed.size() == depth - 1) {
			Node leaf = new Node();
			leaf.setPrediction(majorityOutput);
			return leaf;
		}
		// Find feature that gives best information gain and split on it
		double entropy = getEntropy(data.get(0));
		int index = 0;
		for (int i = 1; i < data.size(); i++) {
			if (!featuresUsed.contains(i)) {
				index = i;
			}
		}
		double maxInfoGain = getInformationGain(data.get(0), data.get(index), entropy);
		for (int i = 1; i < data.size(); i++) {
			if (featuresUsed.contains(i)) {
				continue;
			}
			double infoGain = getInformationGain(data.get(0), data.get(i), entropy);
			if (infoGain > maxInfoGain) {
				maxInfoGain = infoGain;
				index = i;
			}
		}
		featuresUsed.add(index);
		ArrayList<Integer> copyOfFeaturesUsedBefore = new ArrayList<Integer>();
		for (int i = 0; i < featuresUsed.size(); i++) {
			copyOfFeaturesUsedBefore.add(new Integer(featuresUsed.get(i)));
		}
		// Index = numerical representation of which feature to split on
		// Create the root node 
		Node root = new Node(index);
		ArrayList<Double> uniqueValues = getUniqueValues(data.get(index));
		if (uniqueValues.size() > 3) {
			double threshold = getThreshold(data.get(0), data.get(index));
			ArrayList<ArrayList<Double>> newData1 = new ArrayList<>();
			ArrayList<ArrayList<Double>> newData2 = new ArrayList<>();
			for (int i = 0; i < data.size(); i++) {
				newData1.add(new ArrayList<Double>());
				newData2.add(new ArrayList<Double>());
			}
			for (int i = 0; i < data.get(index).size(); i++) {
				if (data.get(index).get(i) < threshold) {
					for (int j = 0; j < newData1.size(); j++) {
						newData1.get(j).add(data.get(j).get(i));
					}
				}
				else {
					for (int j = 0; j < newData1.size(); j++) {
						newData2.get(j).add(data.get(j).get(i));
					}
				
				}
			}
			Node child1 = buildDecisionTree(newData1, featuresUsed, depth);
			// When going down the tree, featuresUsed gets changed so revert it back to the old values
			featuresUsed = new ArrayList<>();
			for (int i = 0; i < copyOfFeaturesUsedBefore.size(); i++) {
				featuresUsed.add(new Integer(copyOfFeaturesUsedBefore.get(i)));
			}
			Node child2 = buildDecisionTree(newData2, featuresUsed, depth);
			root.addChild(child1);
			root.addChild(child2);
			root.setThreshold(threshold);
		}
		else {
			for (double value : uniqueValues) {
				ArrayList<ArrayList<Double>> newData = new ArrayList<>();
				for (int i = 0; i < data.size(); i++) {
					newData.add(new ArrayList<Double>());
				}
				for (int i = 0; i < data.get(index).size(); i++) {
					if (data.get(index).get(i) == value) {
						for (int j = 0; j < newData.size(); j++) {
							newData.get(j).add(data.get(j).get(i));
						}
					}
				}
				// Create a child for each unique value in the feature
				Node child = buildDecisionTree(newData, featuresUsed, depth);
				// When going down the tree, featuresUsed gets changed so revert it back to the old values
				featuresUsed = new ArrayList<>();
				for (int i = 0; i < copyOfFeaturesUsedBefore.size(); i++) {
					featuresUsed.add(new Integer(copyOfFeaturesUsedBefore.get(i)));
				}
				child.setSplitValue(value);
				root.addChild(child);
				root.setThresholdUsed(false);
			}
		}
		return root;
	}
	
	public static double getAverageAge(ArrayList<Double> ages) {
		double total = 0;
		double counter = 0;
		for (double age: ages) {
			if (age != 0) {
				counter++;
				total += age;
			}
		}
		double average = total/counter;
		double ceiling = Math.ceil(average);
		double floor = Math.floor(average);
		if (Math.abs(ceiling - average) > Math.abs(floor - average)) {
			return floor;
		}
		return ceiling;
	}
	
	public static ArrayList<ArrayList<Double>> parseData() {
		ArrayList<ArrayList<Double>> data = new ArrayList<ArrayList<Double>>();
		try {
			URL path = ClassLoader.getSystemResource("titanic.csv");
			File file = new File(path.toURI());
			BufferedReader reader = new BufferedReader(new FileReader(file));
			
			ArrayList<Double> passengerID = new ArrayList<Double>();
			ArrayList<Double> survived = new ArrayList<Double>();
			ArrayList<Double> pClass = new ArrayList<Double>();
			ArrayList<String> name = new ArrayList<String>();
			ArrayList<Double> sex = new ArrayList<Double>();
			ArrayList<Double> age = new ArrayList<Double>();
			ArrayList<Double> sibSp = new ArrayList<Double>();
			ArrayList<Double> parch = new ArrayList<Double>();
			ArrayList<String> ticket = new ArrayList<String>();
			ArrayList<Double> fare = new ArrayList<Double>();
			ArrayList<String> cabin = new ArrayList<String>();
			ArrayList<Double> embarked = new ArrayList<Double>();
			
			boolean firstLine = true;
			String line;
			while((line = reader.readLine()) != null) {
				// Don't need the first line for calculations
				if (firstLine) {
					firstLine = false;
				}
				else {
					// Have to add index 3 and 4 to form name
					String[] stringValues = line.split(",");
					passengerID.add(Double.parseDouble(stringValues[0]));
					survived.add(Double.parseDouble(stringValues[1]));
					pClass.add(Double.parseDouble(stringValues[2]));
					name.add(stringValues[3] + stringValues[4]);
					// Convert gender to numbers
					// male -> 0, female -> 1
					if (stringValues[5].equals("male")) {
						sex.add(0.0);
					}
					else {
						sex.add(1.0);
					}
					// If age is missing put 0 as a placeholder
					if (stringValues[6].equals("")) {
						age.add(0.0);
					}
					else {
						age.add(Double.parseDouble(stringValues[6]));
					}
					sibSp.add(Double.parseDouble(stringValues[7]));
					parch.add(Double.parseDouble(stringValues[8]));
					ticket.add(stringValues[9]);
					fare.add(Double.parseDouble(stringValues[10]));
					cabin.add(stringValues[11]);
					// Convert embarked place to numbers
					// C -> 1, Q -> 2, S -> 3
					if (stringValues.length < 13) {
						embarked.add(1.0);
					}
					else {
						if (stringValues[12].equals("C")) {
							embarked.add(1.0);
						}
						else if (stringValues[12].equals("Q")) {
							embarked.add(2.0);
						}
						else {
							embarked.add(3.0);
						}
					}
				}
			}
			
			double averageAge = getAverageAge(age);
			int index = 0;
			for (double num : age) {
				if (num == 0) {
					age.set(index, averageAge);
				}
				index++;
			}
			
			// Convert ticket to numeric representation
			ArrayList<Double> ticketConverted = new ArrayList<>();
			for (int i = 0; i < ticket.size(); i++) {
				double value = 0;
				for (int j = 0; j < ticket.get(i).length(); j++) {
					value += (int) ticket.get(i).charAt(j);
				}
				ticketConverted.add(value);
			}
			// Convert ticket to numeric representation
			ArrayList<Double> nameConverted = new ArrayList<>();
			for (int i = 0; i < name.size(); i++) {
				double value = 0;
				for (int j = 0; j < name.get(i).length(); j++) {
					value += (int) name.get(i).charAt(j);
				}
				nameConverted.add(value);
			}
			
			data.add(survived);
			data.add(pClass);
			data.add(nameConverted);
			data.add(sex);
			data.add(age);
			data.add(sibSp);
			data.add(parch);
			data.add(ticketConverted);
			data.add(fare);
			data.add(embarked);
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		return data;
	}
	
	
	public static void main(String[] args) {
		ArrayList<ArrayList<Double>> data = parseData();
		// Split the data into training and test data
		ArrayList<ArrayList<Double>> trainingData = new ArrayList<>();
		ArrayList<ArrayList<Double>> testingData = new ArrayList<>();
		for (int i = 0; i < data.size(); i++) {
			trainingData.add(new ArrayList<Double>());
			testingData.add(new ArrayList<Double>());
			//trainingData.add(new ArrayList<Double> (data.get(i).subList(0, (int) Math.floor(60*data.get(i).size()/100))));
			//testingData.add(new ArrayList<Double> (data.get(i).subList(((int) Math.floor(60*data.get(i).size()/100)), data.get(i).size())));
		}
		for (int i = 0; i < trainingData.size(); i++) {
			for (int j = 0; j < (int) Math.floor(60*data.get(i).size()/100); j++) {
				trainingData.get(i).add(new Double(data.get(i).get(j)));
			}
		}
		for (int i = 0; i < testingData.size(); i++) {
			for (int j = (int) Math.floor(60*data.get(i).size()/100); j < data.get(i).size(); j++) {
				testingData.get(i).add(new Double(data.get(i).get(j)));
			}
		}
		// Build the tree using training data
		for (int i = 2; i <= data.size(); i++) {
			
			Node node = buildDecisionTree(trainingData, new ArrayList<Integer>(), i);
			
			double accuracy = calculateAccuracy(node, trainingData);
			double accuracy2 = calculateAccuracy(node, testingData);
			System.out.println("Depth :" + (i - 1));
			System.out.printf("Training: " + "%.2f" + "%%\n", accuracy);
			System.out.printf("Testing: " + "%.2f" + "%%\n", accuracy2);
		}
	}
}
