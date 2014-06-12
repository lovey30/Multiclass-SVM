package peersim.MultiSvm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.core.*;
import jnipegasos.PrimalSVMWeights;
import peersim.cdsim.*;
import peersim.config.Configuration;
import peersim.config.FastConfig;
import peersim.core.*;

public class myNewSVMCode implements CDProtocol {
	
	// New config option to get the learning rate alpha
	 // @config
	 private static final String PAR_ALPHA = "alpha";
	
	 // New config option to get the learning parameter lambda
	 // @config
	 private static final String PAR_LAMBDA = "lambda";

	 //New config option to get the number of iteration
	 // @config
	  private static final String PAR_ITERATION = "iter";
	  
	// Learning parameter 
	protected double lambda;
	
	// Learning rate 
	protected double alpha;
	
	// Linkable identifier */
	protected int lid;
	
	// Number of iteration (T in algorithm)
	public int T;
	private static final String PAR_PROT = "pid";
	protected static int pid;
	private String protocol;
	public static boolean end = false;
	public int num_class;
	public int num_Att;

	public myNewSVMCode(String prefix) {
		lambda = Configuration.getDouble(prefix + "." + PAR_LAMBDA);
		 T= Configuration.getInt(prefix + "." + PAR_ITERATION);
		lid = FastConfig.getLinkable(CommonState.getPid());
		alpha=Configuration.getDouble(prefix + "."+ PAR_ALPHA);
		pid = Configuration.getPid(prefix + "." + PAR_PROT);
	    //protocol = Configuration.getString(prefix + "." + "prot", "local_model1");
	    
	    //protocol= Configuration.getString(prefix + "." + "prot", "local_model");
	}
	
	
	 //Clone an existing instance. The clone is considered new
	 
	public Object clone() {
		myNewSVMCode msvm = null;
		
		try { msvm = (myNewSVMCode)super.clone(); }
		catch( CloneNotSupportedException e ) {} // never happens
		return msvm;
	}
	
	 //private void local_model(Node node,int pid){
//		 
//		 MyNode n =(MyNode) node;
//		 num_Att=n.traindataset.numAttributes()-1;
//		 num_class=n.ClassSet.size();
//	     n.local_sgd = new double[num_Att][num_class];
//	     
//	     System.out.println(num_class);
//	     
//	     n.local_loss_sgd=new double[num_Att][num_class];// the gradient matrix
//	     for (int row = 0; row < num_Att; row ++)
//	     {
//	    	    for (int col = 0; col < num_class; col++)
//	    	    {
//	    	        n.local_loss_sgd[row][col] = 0.0;
//	    	        		
//	    	    }    	
//	     }
//	     
//
//		    double y;
//		    double r = 0.0;
//			//
//			int N = n.traindataset.size();
//			for(int i=0;i<N;i++)
//			{
//				double dot_prod=0.0;
//				
//				double [] wx= new double[num_class];
//				int x_size= n.traindataset.numAttributes()-1;
//				y=n.traindataset.instance(i).classValue();
//				for(int c=0;c<num_class;c++)
//				{
//				
//				  for (int xiter = 0; xiter < x_size; xiter++) 
//				    { //inner dot product loop
//					// input value
//					double xval = n.traindataset.instance(i).value(xiter);
//					// wtvector value
//						double wval = n.wtVec[xiter][c];
//						dot_prod =dot_prod +( xval * wval);
//					}// dot product loop end
//			     wx[c]=dot_prod;
//	           }       
//				double max=0.0;
//				for(int z=0;z<num_class;z++)
//				{
//					if(max<wx[z])
//					{
//						r=z;
//						max=wx[z];
//					}
//				}
//				//for each training example compute the gradient
//				double[][] sgd=new double[x_size][num_class];
//				
//				for(int c=0;c<num_class;c++)
//				{
//				  for (int xiter = 0; xiter < x_size; xiter++) 
//				     { //inner loop input value
//						double xval = n.traindataset.instance(i).value(xiter);
//				        if(c==r)
//					    {
//						    sgd[xiter][c]=xval;
//					    }
//				        else if(c==y)
//				        {
//				        	sgd[xiter][c]=-xval;
//				        }
//				        else
//				        	sgd[xiter][c]=0.0;
//				        
//				        n.local_loss_sgd[xiter][c]=n.local_loss_sgd[xiter][c]+sgd[xiter][c];
//				      }
//				}
//				
//				
//	        }
//			for (int row = 0; row < num_Att; row ++)
//		     {
//		    	    for (int col = 0; col < num_class; col++)
//		    	    {
//			            n.local_loss_sgd[row][col]=(n.local_loss_sgd[row][col]/N);
//			            n.wtVec[row][col]=(n.wtVec[row][col])*lambda;
//	                 }
//	         }
//			for (int row = 0; row < num_Att; row ++)
//		     {
//		    	    for (int col = 0; col < num_class; col++)
//		    	    {
//		    	    	n.local_sgd[row][col]=n.wtVec[row][col]+ n.local_loss_sgd[row][col];
//		    	    }
//		     }
//			double [][] newval=new double[num_Att][num_class];
//			newval=n.local_sgd;
//			// calculating updated weight vector which is w(new)=w(old)-(alpha*(local_sgd))
//			for (int row = 0; row < num_Att; row ++)
//		     {
//		    	    for (int col = 0; col < num_class; col++)
//		    	    {
//		    	    	newval[row][col]=alpha*(n.local_sgd[row][col]);
//		    	    }
//		    }
//			for (int row = 0; row < num_Att; row ++)
//		     {
//		    	    for (int col = 0; col < num_class; col++)
//		    	    {
//		    	    	n.wtVec[row][col]=n.wtVec[row][col]-newval[row][col];
//		    	    	System.out.println(n.wtVec[row][col]);
//		    	    }
//		     }
//			
//	   }
	 //local model build at every node

// protected List<Node> getPeers(Node node) {
//		Linkable linkable = (Linkable) node.getProtocol(lid);
//		if (linkable.degree() > 0) {
//			List<Node> l = new ArrayList<Node>(linkable.degree());			
//			for(int i=0;i<linkable.degree();i++) {
//				l.add( linkable.getNeighbor(i));
//			}
//			return l;
//		}
//		else
//			return null;						
//	}			
	protected Node selectNeighbor(Node node, int pid) {
	// TODO Auto-generated method stub
		 lid=FastConfig.getLinkable(pid);
		Linkable linkable = (Linkable) node.getProtocol(lid);
		if (linkable.degree() > 0) 
			return (Node) linkable.getNeighbor(CommonState.r.nextInt(linkable.degree()));
		else
	return null ;
}
	
	
	
	
	
	//The actual algorithm Implementation

	@Override
	public void nextCycle(Node node1, int pid) {
		// TODO Auto-generated method stub
		MyNode n1 =(MyNode) node1;
		lid=FastConfig.getLinkable(pid);
		Linkable linkable=(Linkable)node1.getProtocol(lid);
		
		
		
		//Gossip with neighbor
		MyNode peer = (MyNode)selectNeighbor(node1, pid);
		System.out.println("Node [" + n1.getID() + "] is gossiping with Node [" + peer.getID() + "]" );
		double[][] local_wtVec = new double[num_Att][num_class];
		n1.local_sgd = new double[num_Att][num_class]; // gradient matrix of function p(w)
	    n1.local_loss_sgd=new double[num_Att][num_class];// the gradient loss matrix over all inputs
	    for (int row = 0; row < num_Att; row ++)
	     {
	    	    for (int col = 0; col < num_class; col++)
	    	    {
	    	        n1.local_loss_sgd[row][col] = 0.0;
	    	        		
	    	    }    	
	     }
        double[][] peer_wtVec=new double [num_Att][num_class];
        local_wtVec=n1.wtVec;
		peer_wtVec=peer.wtVec;
		//Add local wtVec with peer's wtVec
		double[][] update_wtVec = new double[num_Att][num_class];
		System.out.println("weight vector after gossip update: ");
		for (int row = 0; row < num_Att; row ++)
	     {
	    	    for (int col = 0; col < num_class; col++)
	    	    {
	    	    	update_wtVec[row][col]=(local_wtVec[row][col]+peer_wtVec[row][col])/2;	
	    	    }
	     }
		System.out.println("\n");
		//update local wtVec
		n1.wtVec=update_wtVec;
		//Set peer's wtVec also
		peer.wtVec = update_wtVec;
		 double y;
		 double r = 0.0;
		 int N = n1.traindataset.numInstances();
			for(int i=0;i<N;i++)
			{
				double dot_prod=0.0;
				
				double [] wx= new double[num_class];
		
				y=n1.traindataset.instance(i).classValue();
				for(int c=0;c<num_class;c++)
				{
				
				  for (int xiter = 0; xiter < num_Att; xiter++) 
				    { //inner dot product loop
					// input value
					double xval = n1.traindataset.instance(i).value(xiter);
					// wtvector value
						double wval = n1.wtVec[xiter][c];
						dot_prod =dot_prod +( xval * wval);
					}// dot product loop end
			     wx[c]=dot_prod;
	           } 
				double max=0.0;
				for(int z=0;z<num_class;z++)
				{
					if(max<wx[z])
					{
						r=z+1;
						max=wx[z];
					}
				}
				//for each training example compute the gradient
				double[][] sgd=new double[num_Att][num_class];
				
				for(int c=0;c<=num_class-1;c++)
				{
				  for (int xiter = 0; xiter <= num_Att-1; xiter++) 
				     { //inner loop input value
						double xval = n1.traindataset.instance(i).value(xiter);
				        if(c==r-1)
					    {
						    sgd[xiter][c]=xval;
					    }
				        else if(c==y-1)
				        {
				        	sgd[xiter][c]=-xval;
				        }
				        else
				        	sgd[xiter][c]=0.0;
				        n1.local_loss_sgd[xiter][c]=n1.local_loss_sgd[xiter][c]+sgd[xiter][c];
				      }
				}
				
				
	        }
			for (int row = 0; row < num_Att; row ++)
		     {
		    	    for (int col = 0; col < num_class; col++)
		    	    {
			            n1.local_loss_sgd[row][col]=(n1.local_loss_sgd[row][col]/N);
			            n1.wtVec[row][col]=(n1.wtVec[row][col])*lambda;
	                 }
	         }
			for (int row = 0; row < num_Att; row ++)
		     {
		    	    for (int col = 0; col < num_class; col++)
		    	    {
		    	    	n1.local_sgd[row][col]=n1.wtVec[row][col]+ n1.local_loss_sgd[row][col];
		    	    }
		     }
			double [][] newval=new double[num_Att][num_class];
			newval=n1.local_sgd;
			// calculating updated weight vector which is w(new)=w(old)-(alpha*(local_sgd))
			for (int row = 0; row < num_Att; row ++)
		     {
		    	    for (int col = 0; col < num_class; col++)
		    	    {
		    	    	newval[row][col]=alpha*(n1.local_sgd[row][col]);
		    	    }
		    }
			for (int row = 0; row < num_Att; row ++)
		     {
		    	    for (int col = 0; col < num_class; col++)
		    	    {
		    	    	n1.wtVec[row][col]=n1.wtVec[row][col]-newval[row][col];
		    	    	System.out.println("The new weight matrix "+ n1.wtVec[row][col]+"\n");
		    			
		    	    }
		     }
			for (int row = 0; row < num_Att; row ++)
		     {
				for(int col=0;col< num_class;col++)
				{
				   n1.frobenius_norm=n1.frobenius_norm+Math.pow(n1.wtVec[row][col]);
	             }
			}
			System.out.println("The final norm:"+Math.sqrt(n1.frobenius_norm));
	 }
	 
//		
//	private void flooding1(Node node, int pid){
//		// Gossip with a neighbor
//	   MyNode n= (MyNode) node;
//	   HashSet UpdateSet=new HashSet(n.ClassSet);
//		for(int l=0;l<10;l++){ //loop the 
//				
//	
//		MyNode peer = (MyNode)selectNeighbor(node, pid);
//		System.out.println("Node [" + n.getID() + "] is gossiping with Node [" + peer.getID() + "]" );
//		
//		HashSet PeerClassSet=new HashSet();
//		PeerClassSet=peer.ClassSet;
//	//add local classes with peer classes
//		
//	    UpdateSet.add(peer.ClassSet);
//		n.ClassSet=UpdateSet;
//		peer.ClassSet=UpdateSet;
//		}
//		num_class=UpdateSet.size();
// }
 

 
 
} 
		
		
	
	



