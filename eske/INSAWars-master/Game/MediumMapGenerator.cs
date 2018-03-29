
namespace INSAWars.Game
{
    public class MediumMapGenerator : MapGenerator
    {
        private const int SIZE = 100;

        public override Map generate(MapConfiguration config)
        {
            return generate(config, SIZE);
        }
    }
}
