import java.util.ArrayList;

public class Node {
	private int feature;
	private ArrayList<Node> children = new ArrayList<>();
	private double prediction;
	private boolean leaf = false;
	private boolean thresholdUsed;
	private double threshold;
	private double splitValue;
	
	
	public Node() {}
	
	public Node(int feature) {
		this.feature = feature;
		this.leaf = false;
	}
	
	public int getFeature() {
		return this.feature;
	}
	
	public ArrayList<Node> getChildren() {
		return this.children;
	}
	
	public double getPrediction() {
		return this.prediction;
	}
	
	public boolean isLeaf() {
		return this.leaf;
	}
	
	public boolean thresholdUsed() {
		return thresholdUsed;
	}
	
	public double getThreshold() {
		return this.threshold;
	}
	
	public double getSplitValue() {
		return this.splitValue;
	}
	
	public void setPrediction(double prediction) {
		this.prediction = prediction;
		this.leaf = true;
	}
	 
	public void setThresholdUsed(boolean value) {
		this.thresholdUsed = value;
	}
	
	public void setThreshold(double threshold) {
		this.thresholdUsed = true;
		this.threshold = threshold;
	}
	
	public void setSplitValue(double splitValue) {
		this.splitValue = splitValue;
	}
	
	public void addChild(Node child) {
		this.children.add(child);
	}
	
}
