using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWrapper;

namespace INSAWars.Game
{
    public abstract class MapGenerator
    {
        public abstract Map generate(MapConfiguration config);

        public virtual Map generate(MapConfiguration config, int size)
        {
            PerlinMapWrapper perlinMap = new PerlinMapWrapper(size, config.octaves, config.persistance, config.terrains, config.decorators);
            Case[,] cases = new Case[size, size];
            List<Case> startingPositions = new List<Case>();

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    int terrainIndex = perlinMap.GetTerrain(i, j);
                    int decoratorIndex = perlinMap.GetDecorator(i, j);
                    cases[i, j] = MapConfiguration.GetCase(terrainIndex, decoratorIndex, i, j);
                }
            }

            return new Map(cases);
        }
    }
}
