
namespace INSAWars.Game
{
    public class SmallMapGenerator : MapGenerator
    {
        private const int SIZE = 25;

        public override Map generate(MapConfiguration config)
        {
            return generate(config, SIZE);
        }
    }
}
