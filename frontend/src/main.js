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
        const angle = this.heading * Math.PI / 180
        return [Math.sin(angle), Math.cos(angle)]
    }

    move(seconds = 1){
        const e = this.get_direction_vector()
        this.x += e[0] * this.speed * seconds / 3600
        this.y += e[1] * this.speed * seconds / 3600
    }

    create_position(rel_direction, distance, heading, speed){
        const angle = (this.heading + rel_direction) * Math.PI / 180
        return new Position(
            this.x + distance * Math.sin(angle),
            this.y + distance * Math.cos(angle),
            heading,
            speed)
    }

    get_relative_position(pos){
        const dx = pos.x - this.x
        const dy = pos.y - this.y
        const distance = Math.sqrt(dx*dx + dy*dy)
        const angle = 90 - Math.atan2(dy, dx) * 180 / Math.PI
        const rel_direction = (this.heading - angle) % 360
        return {distance, rel_direction}
    }

    to_svg_path(size=0.4){
        const efwd_x = Math.sin(this.heading*Math.PI / 180)
        const efwd_y = Math.cos(this.heading*Math.PI / 180)
        const enrm_x = efwd_y
        const enrm_y = -efwd_x

        const tip_x = this.x + efwd_x * size * 2.5
        const tip_y = this.y + efwd_y * size * 2.5
        const base1_x = this.x + enrm_x * size - efwd_x * size * 1.5
        const base1_y = this.y + enrm_y * size - efwd_y * size * 1.5
        const base2_x = this.x - enrm_x * size - efwd_x * size * 1.5
        const base2_y = this.y - enrm_y * size - efwd_y * size * 1.5
        return "M " + tip_x + " " + tip_y + " L " + base1_x + " " + base1_y + " L " + base2_x + " " + base2_y + " Z"
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

    is_warship(){
        return (this.ship_type != "Unbekannt") && (this.ship_type != 'Handelsschiff')
    }
}

function increment_ship_id(state){
    state.ship_id_counter++
    return state.ship_id_counter
}

function add_ship(state, ship_data) {

    const pos = state.uboot_pos.create_position(
        ship_data.rel_direction,
        ship_data.distance,
        ship_data.heading,
        ship_data.speed)
    const ship = new Ship(
        increment_ship_id(state),
        ship_data.ship_type,
        ship_data.name,
        ship_data.tons,
        pos
        )
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
        uboot_pos: new Position(0,0,0,0),
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
