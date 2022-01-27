import Vue from 'vue'
import Vuex from 'vuex'
import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'
import App from './App.vue'
import BootstrapVue from "bootstrap-vue";

Vue.config.productionTip = false
Vue.use(BootstrapVue)
Vue.use(Vuex)

class Position{
    constructor(x,y,heading, speed)
    {
        this.x = x
        this.y = y
        this.heading = heading
        this.speed = speed
    }

    get_direction_vector(){
        const angle = this.heading * 2 * Math.PI
        return [Math.sin(angle), Math.cos(angle)]
    }

    move(seconds = 1){
        const e = this.get_direction_vector()
        this.x += e[0] * this.speed * seconds
        this.y += e[1] * this.speed * seconds
    }
}

// eslint-disable-next-line no-unused-vars
class Ship{
    constructor(id=0, ship_type = "Unbekannt", name = "Kontakt", tons = 0, pos=null, alive=true) {
        if (pos == null){
            pos = new Position(0, 0, 0, 0)
        }
        this.id = id
        this.alive = alive
        this.ship_type = ship_type
        this.name = name
        this.tons = tons
        this.pos = pos
    }
}

function increment_ship_id(state){
    state.ship_id_counter++
    return state.ship_id_counter
}

function add_ship(state, ship_data) {
    let ship = new Ship(increment_ship_id(state))
    ship.name = ship_data.name
    ship.tons = ship_data.tons
    ship.ship_type = ship_data.ship_type
    ship.pos.speed = ship_data.speed
    state.ships.push(ship)
}

function sink_ship(state, id) {
    state.ships.find(s => s.id == id).alive = false
}

function remove_ship(state, id) {
    console.log("Removing ship: " + id)
    state.ships = state.ships.filter(s => s.id != id)
    console.log(state.ships.length)
}


const store = new Vuex.Store({
    state: {
        ship_id_counter: 0,
        time: 0,
        ships: []
    },
    mutations: {
        add_ship: add_ship,
        sink_ship: sink_ship,
        remove_ship: remove_ship,
        pass_time: function(state, seconds=1){
            state.time += seconds
            state.ships.map((s) => s.pos.move(seconds))
        }
    }
})

setInterval(() => store.commit('pass_time'), 1000)

new Vue({
    render: h => h(App),
    store: store
}).$mount('#app')
