#region usings
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Units;
using INSAWars.MVVM;
using System.ComponentModel;
#endregion

namespace INSAWars.Game
{

    [Serializable]
    public class City : ObservableObject
    {
        #region fields
        public const int Radius = 2;

        private List<Case> territory;
        private List<Case> fields;
        private Case position;
        private Player player;
        private int _population;
        
        private int _food;
        private int _requiredFood;
        private int _iron;

        private List<Unit> pendingProductions;
        #endregion

        #region properties
        public int Food
        {
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

        public Player Player
        {
            get { return player; }
        }

        public Case Position
        {
            get { return position; }
        }

        private AbstractUnitFactory Factory
        {
            get { return player.Civilization.UnitFactory; }
        }
        #endregion

        #region constructors
        /// <summary>
        /// Creates a new city on a given case.
        /// </summary>
        /// <param name="position">The position of the city.</param>
        /// <param name="player">The player to whom the city belongs.</param>
        /// <param name="name">The name of the city (defined by the player).</param>
        public City(Case position, Player player, List<Case> territory)
        {
            this.position = position;
            this.player = player;
            this.fields = new List<Case>();
            this.fields.Add(this.position);
            this.pendingProductions = new List<Unit>();
            this.territory = territory;
            this.territory = territory.OrderBy(item => item.DistanceTo(position)).ToList();
            _population = 1;
            _food = 0;
            _iron = 0;
            _requiredFood = 10;
        }
        #endregion

        #region methods
        public void Invade(Player invader)
        {
            player.RemoveCity(this);
            invader.AddCity(this);
            player = invader;

            foreach (Case field in fields.ToList())
            {
                if (field.HasUnits)
                {
                    // The case is occupied by the enemy and cannot be used as a field.
                    RemoveField(field);
                    field.Free();
                }
                else
                {
                    // Changes the occupant of the field.
                    field.Use(this);
                }
            }
        }

        public void CancelProduction(int index)
        {
            pendingProductions.RemoveAt(index);
        }

        public void MakeStudent()
        {
            Student unit = player.Civilization.UnitFactory.CreateStudent(position, player);
            pendingProductions.Add(unit);
            Food -= Factory.StudentFoodCost;
            Iron -= Factory.StudentIronCost;
        }

        public bool CanMakeStudent()
        {
            return _food >= Factory.StudentFoodCost && _iron >= Factory.StudentIronCost;
        }

        public void MakeTeacher()
        {
            Teacher unit = player.Civilization.UnitFactory.CreateTeacher(position, player);
            pendingProductions.Add(unit);
            Food -= Factory.TeacherFoodCost;
            Iron -= Factory.TeacherIronCost;
        }

        public bool CanMakeTeacher()
        {
            return _food >= Factory.TeacherFoodCost && _iron >= Factory.TeacherIronCost;
        }

        public void MakeHead()
        {
            Head unit = player.Civilization.UnitFactory.CreateHead(position, player);
            pendingProductions.Add(unit);
            Food -= Factory.HeadFoodCost;
            Iron -= Factory.HeadIronCost;
            player.Head = unit;
        }

        public bool CanMakeHead()
        {
            return player.Head == null && _food >= Factory.HeadFoodCost && _iron >= Factory.HeadIronCost;
        }

        public void Expand()
        {
            List<Case> freeTerritory = new List<Case>();
            int distance = 0;

            foreach (Case c in territory)
            {
                if (c.IsFree && (!c.HasUnits || c.Occupant == Player))
                {
                    if (distance == 0)
                    {
                        distance = c.DistanceTo(position);
                    }

                    // Use in priority the closest cases.
                    if (c.DistanceTo(position) == distance)
                    {
                        freeTerritory.Add(c);
                    }
                }
            }

            if (freeTerritory.Count > 0)
            {
                int r = Game.random.Next(freeTerritory.Count);
                Case field = freeTerritory[r];

                fields.Add(field);
                field.Use(this);

                Population++;
                Food -= _requiredFood;
                _requiredFood += _requiredFood / 2;
            }
        }

        public void RemoveField(Case field)
        {
            Population--;
            _requiredFood = _requiredFood * 2 / 3;
            fields.Remove(field);

        }

        public bool CanExpand()
        {
            return _food >= _requiredFood && fields.Count < territory.Count;
        }

        public void NextTurn()
        {
            CollectResources();
            HandleProduction();

            if (CanExpand())
            {
                Expand();
            }
        }

        private void HandleProduction()
        {
            foreach (Unit unit in pendingProductions)
            {
                player.AddUnit(unit);
                position.AddUnit(unit);
            }

            pendingProductions.Clear();
        }

        private void CollectResources()
        {
            foreach (Case c in fields) {
                _food += c.Food;
                _iron += c.Iron;
            }
        }

        public void Destroy()
        {
            position.Free();
            player.RemoveCity(this);
        }

        public override string ToString()
        {
            return "City of [" + Player + "] at coordinates [" + position.X + ", " + position.Y + "]";
        }
        #endregion
    }
}
