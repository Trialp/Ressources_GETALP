using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWrapper;
using INSAWars.Units;
using INSAWars.Units.Info;
using INSAWars.Game;

namespace INSAWars.Game
{
    /// <summary>
    /// Defines a Builder Pattern to build a new game step by step.
    /// </summary>
    public class GameBuilder
    {
        private MapGenerator mapGenerator;
        private Dictionary<string, ICivilization> players;
        private MapConfiguration mapConfig;

        /// <summary>
        /// Creates a new game builder.
        /// You have to define the settings by calling AddPlayer, SetSize and SetTerrainFrequency and SetDecoratorProbability, as there are not default settings.
        /// </summary>
        public GameBuilder()
        {
            mapConfig = new MapConfiguration();
            players = new Dictionary<string, ICivilization>();
        }

        /// <summary>
        /// Defines the size of the map, either small or medium.
        /// </summary>
        /// <param name="size">Either "Small" for a 25x25 map, or "Medium" for a 100x100</param>
        public void SetSize(string size)
        {
            if (size == "Small")
            {
                mapGenerator = new SmallMapGenerator();
            }
            else
            {
                mapGenerator = new MediumMapGenerator();
            }
        }

        /// <summary>
        /// Adds a new player to the settings.
        /// The number of players has to be between 2 and 4.
        /// </summary>
        /// <param name="name">Name of the player</param>
        /// <param name="civ">Civilization of the player, either "INFO" or "EII".</param>
        public void AddPlayer(string name, string civ)
        {
            players.Add(name, CivilizationFactory.GetCivilizationByName(civ));
        }

        /// <summary>
        /// Sets the frequency of the given terrain.
        /// The sum of the frequencies must be 1.
        /// </summary>
        /// <param name="terrainIndex">Enum Terrain</param>
        /// <param name="frequency">The desired frequency of the given terrain on the map</param>
        public void SetTerrainFrequency(int terrainIndex, double frequency)
        {
            mapConfig.terrains[terrainIndex] = frequency;
        }

        /// <summary>
        /// Substitute to SetDecoratorProbability and SetTerrainFrequency.
        /// </summary>
        public void UseDefaultFrequencies()
        {
            mapConfig.terrains[(int) Terrains.DESERT] = 0.3;
            mapConfig.terrains[(int) Terrains.PLAIN] = 0.5;
            mapConfig.terrains[(int) Terrains.MOUNTAIN] = 0.2;
            mapConfig.terrains[(int)Terrains.WATER] = 0;
            mapConfig.decorators[(int) Decorators.FOOD] = 0.2;
            mapConfig.decorators[(int) Decorators.IRON] = 0.2;
        }

        /// <summary>
        /// Sets the probability for each case to have the given decorator.
        /// The sum of the probability may not exceed 1.
        /// </summary>
        /// <param name="decoratorIndex">Enum Decorator</param>
        /// <param name="probability">Probability for each case of having the given decorator</param>
        public void SetDecoratorProbability(int decoratorIndex, double probability)
        {
            mapConfig.decorators[decoratorIndex] = probability;
        }

        /// <summary>
        /// Builds a new game. Builds the map and initializes the players according to the chosen settings.
        /// </summary>
        /// <returns>A new game, ready to run</returns>
        public Game Build()
        {
            Map map = mapGenerator.generate(mapConfig);

            List<Player> players = new List<Player>();

            foreach (var entry in this.players)
            {
                Player player = new Player(entry.Value, entry.Key);
                players.Add(player);
                initPlayer(player, map);                
            }

            return new Game(map, players);
        }

        /// <summary>
        /// Restores a previously saved game from the save file.
        /// </summary>
        /// <param name="filename">Location of the save file</param>
        /// <returns>The restored game.</returns>
        public static Game LoadGame(string filename)
        {
            Game game = null;
            Stream stream = File.Open(filename, FileMode.Open);
            BinaryFormatter formatter = new BinaryFormatter();

            game = (Game)formatter.Deserialize(stream);
            stream.Close();

            return game;
        }

        /// <summary>
        /// Saves the current game state into the given file.
        /// </summary>
        /// <param name="filename">Location of the save file.</param>
        public static void SaveGame(string filename, Game g)
        {
            Stream stream = File.Open(filename, FileMode.Create);
            BinaryFormatter formatter = new BinaryFormatter();
            formatter.Serialize(stream, g);
            stream.Close();
        }

        /// <summary>
        /// Initializes a player with default values : a teacher and a student.
        /// </summary>
        /// <param name="player"></param>
        /// <param name="map"></param>
        private void initPlayer(Player player, Map map)
        {
            Case position = map.FreePosition;
            Teacher teacher = player.Civilization.UnitFactory.CreateTeacher(position, player);
            Student student = player.Civilization.UnitFactory.CreateStudent(position, player);
            player.AddUnit(teacher);
            position.AddUnit(teacher);
            player.AddUnit(student);
            position.AddUnit(student);
        }
    }
}
