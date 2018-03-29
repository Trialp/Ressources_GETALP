using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using INSAWars.Units;

namespace INSAWars.Game
{
    [Serializable]
    public class Map
    {
        private Case[,] cases;
        private Stack<Case> startingPositions;

        public int Size
        {
            get { return cases.GetLength(0); }
        }

        public Case GetCaseAt(int x, int y)
        {
            return cases[x, y];
        }

        /// <summary>
        /// Returns the next free case.
        /// </summary>
        public Case FreePosition
        {
            get { return startingPositions.Pop(); }
        }

        /// <summary>
        /// Gives the cases around the given position, at the given maximum distance.
        /// </summary>
        /// <param name="position">Center position.</param>
        /// <param name="distance">Maximum distance from the center position.</param>
        /// <returns></returns>
        public List<Case> TerritoryAround(Case position, int distance)
        {
            List<Case> territory = new List<Case>();

            for (int x = Math.Max(0, position.X - distance); x <= Math.Min(Size - 1, position.X + distance); x++)
            {
                for (int y = Math.Max(0, position.Y - distance); y <= Math.Min(Size - 1, position.Y + distance); y++)
                {
                    Case c = GetCaseAt(x, y);

                    if (!(c is Water))
                    {
                        territory.Add(GetCaseAt(x, y));
                    }
                }
            }

            return territory;
        }

        /// <summary>
        /// Great algorithm originally asked to be written in C++.
        /// Finds the 3 best cases where to build a city for the given teacher,
        /// according to the location of resources and enemy units.
        /// TODO: test the algorithm to see if the weights are well-balanced...
        /// </summary>
        /// <param name="teacher">Teacher who builds the city.</param>
        /// <returns>A list of the best 3 cases.</returns>
        public List<Case> BestCases(Teacher teacher)
        {
            var casesWeights = new List<KeyValuePair<Case, double>>();

            foreach (Case c in cases)
            {
                if (c.HasCity || (c.Occupant != null && c.Occupant != teacher.Player))
                {
                    // The case is occupied by an enemy, or a city is already built on it.
                    continue;
                }

                double weight = 0;

                // Check the status of each field of the city to be built.
                foreach (Case field in TerritoryAround(c, City.Radius))
                {
                    if (c.Occupant != null && c.Occupant != teacher.Player)
                    {
                        // If one of the fields contains enemy units.
                        weight -= 10;
                    }
                    else if (c.HasCity)
                    {
                        // If the current player has a city on one of the fields
                        // (the two cities are too close).
                        weight -= 5;
                    }
                    else if (c.IsUsed)
                    {
                        // The field is already used by another city.
                        weight -= 2;
                    }
                    else
                    {
                        // The player has units on this case ready to defend it.
                        weight += field.Units.Count();

                        // Of course, the resources count.
                        weight += field.Food;
                        weight += field.Iron;
                    }
                }

                // Divides the weight by the number of turns needed for the teacher to reach the location.
                weight /= 1 + c.DistanceTo(teacher.Location) / (teacher.RemainingMovementPoints + 1);

                if (weight > 0)
                {
                    casesWeights.Add(new KeyValuePair<Case, double>(c, weight));
                }
            }

            // Sort by best weight.
            casesWeights.Sort((a, b) => a.Value.CompareTo(b.Value));

            var bestCases = new List<Case>();

            // Keep only the 3 best ones.
            for (int i = 0; i < 3 && i < casesWeights.Count; i++)
            {
                bestCases.Add(casesWeights[i].Key);
            }

            return bestCases;
        }

        /// <summary>
        /// Creates a new map with the given cases array.
        /// </summary>
        /// <param name="cases">2D array of cases.</param>
        public Map(Case[,] cases)
        {
            this.cases = cases;
            startingPositions = new Stack<Case>();

            // Initialize the player positions (up to 4 players) at the corners of the map.
            startingPositions.Push(RandomCase(cases[0, 0], 5));
            startingPositions.Push(RandomCase(cases[Size - 1, Size - 1], 5));
            startingPositions.Push(RandomCase(cases[Size - 1, 0], 5));
            startingPositions.Push(RandomCase(cases[0, Size - 1], 5));
        }

        /// <summary>
        /// Returns a random case in the given region.
        /// The region is delimited by a center position a maximum distance to this position.
        /// </summary>
        /// <param name="position">Center position.</param>
        /// <param name="distance">Maximum distance.</param>
        /// <returns></returns>
        private Case RandomCase(Case position, int distance)
        {
            List<Case> cases = TerritoryAround(position, distance);
            return cases[Game.random.Next() % cases.Count];
        }

        public override string ToString()
        {
            var builder = new StringBuilder("Map: \n");
            
            foreach (Case _case in cases)
            {
                builder.AppendLine(_case.ToString());
            }

            return builder.ToString();
        }
    }
}
