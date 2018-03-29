using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using INSAWars.Game;
using INSAWars.MVVM;
using System.ComponentModel;
using INSAWars.Units;

namespace UI.Views
{
    public class CityView : ObservableObject
    {
        private City _city;
        private int _food;
        private int _iron;
        private int _population;

        public CityView(City c)
        {
            _city = c;

            c.PropertyChanged += new PropertyChangedEventHandler(delegate(object sender, PropertyChangedEventArgs args)
            {
                var city = (City) sender;

                switch (args.PropertyName)
                {
                    case "Food":
                        Food = city.Food;
                        break;
                    case "Iron":
                        Iron = city.Iron;
                        break;
                    case "Population":
                        Population = city.Population;
                        break;
                    case "CanMakeStudent":
                        break;
                    case "CanMakeHead":
                        break;
                    case "CanMakeTeacher":
                        break;
                    default:
                        break;
                }
            });

            Food = c.Food;
            Iron = c.Iron;
            Population = c.Population;
        }

        public int Food {
            get { return _food; }
            set
            {
                SetProperty(ref _food, value);
            }
        }

        public int Iron
        {
            get { return _iron; }
            set
            {
                SetProperty(ref _iron, value);
            }
        }

        public int Population
        {
            get { return _population; }
            set
            {
                SetProperty(ref _population, value);
            }
        }
    }
}
