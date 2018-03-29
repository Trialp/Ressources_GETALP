using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using INSAWars.Game;

namespace INSATest
{
    class Program
    {
        static void Main(string[] args)
        {
            bool save = false;

            GameBuilder builder = new GameBuilder();

            if (save)
            {
                builder.AddPlayer("Roger", "INFO");
                builder.AddPlayer("Bernard", "EII");
                builder.SetSize("Small");
                builder.UseDefaultFrequencies();
                Game game = builder.Build();
                game.Save("save01.dat");
            }
            else
            {
                Game game = builder.LoadGame("save01.dat");
                //Console.Write(game.Map);
                Console.WriteLine(game.CurrentPlayer);
                Console.WriteLine(game.CurrentPlayer.Cities[0]);
                game.NextTurn();
                Console.WriteLine(game.CurrentPlayer);
                Console.WriteLine(game.CurrentPlayer.Cities[0]);
                game.Save("save01.dat");
            }
        }
    }
}
