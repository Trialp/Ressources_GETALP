using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace INSAWars.Game
{
    /// <summary>
    /// Defines a map configuration in order to generate different kind of maps (more or less plains for example).
    /// </summary>
    public class MapConfiguration
    {
        public int octaves = 6;
        public double persistance = 0.5;
        public double[] terrains;
        public double[] decorators;
        public static int[] inaccessibleTerrains = new int[] { (int) Terrains.WATER };

        public MapConfiguration()
        {
            terrains = new double[4];
            Array.Clear(terrains, 0, terrains.Length);
            decorators = new double[2];
            Array.Clear(decorators, 0, decorators.Length);
        }

        public static Case GetCase(int terrainIndex, int decoratorIndex, int x, int y)
        {
            switch (terrainIndex)
            {
                case (int)Terrains.WATER:
                    return new Water(x, y);
                case (int)Terrains.DESERT:
                    return GetDecorator(decoratorIndex, new Desert(x, y));
                case (int)Terrains.MOUNTAIN:
                    return GetDecorator(decoratorIndex, new Mountain(x, y));
                default:
                    return GetDecorator(decoratorIndex, new Plain(x, y));
            }
        }

        private static Case GetDecorator(int decoratorIndex, Case decoratedCase)
        {
            switch (decoratorIndex)
            {
                case (int)Decorators.IRON:
                    return new IronCaseDecorator(decoratedCase);
                case (int)Decorators.FOOD:
                    return new FoodCaseDecorator(decoratedCase);
                default:
                    return decoratedCase;
            }
        }
    }

    public enum Terrains
    {
        WATER = 0,
        DESERT = 1,
        PLAIN = 2,
        MOUNTAIN = 3
    }

    public enum Decorators
    {
        FOOD = 0,
        IRON = 1
    }
}
