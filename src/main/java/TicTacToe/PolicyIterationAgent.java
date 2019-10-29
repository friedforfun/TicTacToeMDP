package TicTacToe;

import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{
		//initialise this.curPolicy to a random policy
		//for all games, set the policy to return a random action possible from that game.
		
		Random r = new Random(new Date().getTime());
		//iterate over all games
		
		for(Game g: this.policyValues.keySet())
		{
			List<Move> possibleMoves = g.getPossibleMoves();
			//if game is terminal, no need to have a policy entry for it.
			if (g.isTerminal())
				continue;
			
			//pick random move for g, and put it into the policy map.
			this.curPolicy.put(g, possibleMoves.get(r.nextInt(possibleMoves.size())));
			
			
		}
	}
	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@param delta}, in other words
	 * until the values under the currrent policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)
	{
		/* YOUR CODE HERE */
		
		System.out.println("Evaluating current policy");
		
		/**
		 * maxUpdate stores the current maximum update we are making to the value of states under the current policy.
		 * this is to check for convergence. If this maximum update is less than delta, the game/state values
		 * under the current policy are not changing much. So we stop.
		 * We take this to always be a positive number, i.e. the absolute value of the difference between the previous value
		 * and the current updated value.
		 */
		double maxUpdate=0;
		
		//a fresh hashmap to store the new values for the current policy as it is being evaluated.
		
		HashMap<Game, Double> newValues=new HashMap<Game, Double>(); 
		int i=1;
		do {
			
			maxUpdate=0;
			
			System.out.println("Step "+i);
			
		
			//iterate over all games to calcluate their value under curPolicy
			//The difference from the Value Iteration algorithm is that there is no maxing over actions/moves
			
			for(Game g: policyValues.keySet())
			{
				double sum=0.0;
				
				/**
				 * if g is terminal just set its value to 0 - like in value iteration.
				 */
				if (g.isTerminal())
				{
					newValues.put(g, 0.0);
					continue;
				}
				
				/**
				 * This is exactly the same as the way we calculate the sum for Value Iteration. Except that
				 * here the action/move taken is that which the current Policy tells us to take, rather than
				 * trying all actions for find the maximum.
				 */
				List<TransitionProb> transitions=this.mdp.generateTransitions(g, curPolicy.get(g));
				
				for(TransitionProb tp:transitions)
				{
					sum+=tp.prob*(tp.outcome.localReward + this.discount*this.policyValues.get(tp.outcome.sPrime));
					
				}
				
				//the absolute value of the update, i.e. the difference between the previous value of the current policy
				//and the new value after our evaluation step
				
				double update=Math.abs(policyValues.get(g) - sum);
				
				/**
				 * Is the new update size more than the current maximum update size?
				 */
				if (update>=maxUpdate)
				{
					
					maxUpdate=update;
				}
				//now update the current policy values
				newValues.put(g, sum);
				
				
				
			
			
			}
			
			System.out.println("Max update was:"+maxUpdate);
			//overwrite the current policy values with new values
			this.policyValues=newValues;
			i++;
			
		} while(maxUpdate>delta);
		
		/**
		 * The above loop will run until the maximum update to the value function is less than the small delta. 
		 * It means that the values are no longer updating much, and that we have converged. 
		 */
	}
		
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#valueFuncion}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
		/* YOUR CODE HERE */
		System.out.println("Improving policy");
		
		//keeping track of whether policy changed
		boolean policyChanged=false;
		
		for(Game g: curPolicy.keySet())
		{
			
			
			Move maxMove=null;
			double maxValue=-999;
			
			for(Move m: g.getPossibleMoves())
			{
				double sum=0;
				List<TransitionProb> transitions=mdp.generateTransitions(g, m);
				for(TransitionProb tp: transitions)
					sum+=tp.prob*(tp.outcome.localReward + policyValues.get(tp.outcome.sPrime));
				
				//we break ties/equality of values here by checking a >= instead of a >
				if (sum>=maxValue)
				{
					maxMove=m;
					maxValue=sum;
				}
				
			}
			
			if (!maxMove.equals(curPolicy.get(g)))
			{
				//update policy if the value maximising move is not the same as the current policy's choice of move
				curPolicy.put(g, maxMove);
				//If we are here, policy is till changing.
				policyChanged=true;
			}
			
			
		}
		
		return policyChanged;
	}
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		/* YOUR CODE HERE */
		
		//evaluate and improve policy while the policy is changing, i.e. while improvePolicy returns true. 
		do {
			evaluatePolicy(delta);
			
		}while(improvePolicy());
		
		this.policy=new Policy(this.curPolicy);
		
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the evaluatePolicy(), improvePolicy() & train() methods");
			System.exit(1);
		}
		
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}
