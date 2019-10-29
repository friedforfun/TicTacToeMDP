package ticTacToe;


import java.util.HashMap;

/**
 * A Q-Learning agent with a Q table, i.e. a table of Q-Values. This tables is implemented as a {@link java.util.HashMap} from {@link Game}
 * objects to another HashMap from Moves to Q-Values. It could be implemented in many other ways too, but I thought this seemed the simplest.
 *  
 *  
 * @author ae187
 *
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.2;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=100;
	
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair, you can do
	 * qTable.get(game).get(move) which return the Q(game,move) value stored. Be careful with 
	 * cases where there is currently no value. You can use the containsKey method to check if the mapping is there.
	 * 
	 */
	
	HashMap<Game, HashMap<Move, Double>> qTable=new HashMap<Game, HashMap<Move, Double>>();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration & policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()
	{
		
	}
	
	
	/**
	 * 
	 */
	public void train()
	{
		
		for(int i=0;i<this.numEpisodes;i++)
		{
			
			System.out.println("---------");
			System.out.println("Episode "+i);
			
			
			
			
			
			
		}
		
		
		
	}
	
	
	
	


	
}
