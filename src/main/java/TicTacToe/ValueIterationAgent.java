package TicTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=10;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	public void iterate()
	{
		//perform k steps of value iteration
		for(int i=0;i<this.k;i++) {
			
			//initialise a fresh value function for the next step of the iteration
			Map<Game, Double> next=new HashMap<Game, Double>();
			
			
			//iterate over ALL the games in valueFunction
			for(Game g: valueFunction.keySet())
			{
				
				// we need to iterate over all the possible moves from game g to get the maximum value possible from g
				//set max to a very low value - this is the standard way of finding the maximum value of a set.
				
				double max=-999.00;
				
				
				//If g is terminal, then its value must always and for all iterations be 0. No other calculations needed.
				//Note: the agent would have received its reward already when it transitions into a terminal state, so taht
				//the reward is REFLECTED IN THE VALUE OF PRE-TERMINAL STATES
				if (g.isTerminal())
				{
					next.put(g,new Double(0.0));
					continue;
				}
				List<Move> possibleMoves=g.getPossibleMoves();
				
				//now iterate over all moves {@code possibleMoves} to find the maximum value.
				for(Move m: possibleMoves)
				{
					//performing move m in state g, now average the future rewards
					//iterate over transitions to find the value of the sum in the Bellman Equations
					List<TransitionProb> transitions=this.mdp.generateTransitions(g, m);
					//initialise sum to 0
					double sum=0.0;
					
					for(TransitionProb tp: transitions)
					{
						//this is the main thrust of the calculation, coming from inside the Bellman Equations
						//the right hand side of this update corresponds to: 
						//T(s, a, s') * [R(s,a,s') + gamma * V(s')
						//see Value Iteration Update in the lecture notes..
						
						sum+=tp.prob*(tp.outcome.localReward + this.discount*this.valueFunction.get(tp.outcome.sPrime));
					
					}
					if (sum>max)
						max=sum;
					
				}
				//at this point we have the maximum (optimal value) for game g, i.e. V*(g)
				//write it into the hashmap
				next.put(g, max);
				
				
			}
			
			//At this point, we have computed the ith iteration fully, and have it in {@code next}
			//now overwrite the previous valueFunction
			valueFunction=next;
			
			
		}
		
	}
	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		//for each game, compute the average future value of making move m - i.e. calculate Q(g,m)
		//set the policy to return the m with maximal Q(g,m)
		
		Policy result=new Policy();
		for(Game g: this.valueFunction.keySet())
		{
			
			//if g is terminal, the policy has nothing to say; just continue.
			if (g.isTerminal())
				continue;
			
			Move maxMove=null;
			double max=-999.00;
			for(Move m: g.getPossibleMoves())
			{
				double sum=0.0;
				List<TransitionProb> transitions=this.mdp.generateTransitions(g, m);
				for(TransitionProb tp: transitions)
				{
					sum+=tp.prob*(tp.outcome.localReward + this.discount * this.valueFunction.get(tp.outcome.sPrime));
				}
				if (sum>=max)
				{
					max=sum;
					maxMove=m;
				}
				
			}
			//System.out.println("For "+g);
			//System.out.println("Policy says:"+maxMove);
			if (maxMove==null)
			{
				System.out.println("Max move is null. This shouldn't happen");
				System.exit(1);
			}
			result.policy.put(g, maxMove);
			
		}
		
		return result;
	}
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
}
