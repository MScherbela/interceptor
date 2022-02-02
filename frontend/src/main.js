import Vue from 'vue'
import Vuex from 'vuex'
import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'
import App from './App.vue'
import {BootstrapVue, BootstrapVueIcons} from "bootstrap-vue";

Vue.config.productionTip = false
Vue.use(BootstrapVue)
Vue.use(BootstrapVueIcons)
Vue.use(Vuex)

class Position {
    constructor(x, y, heading, speed) {
        this.x = x
        this.y = y
        this.heading = heading
        this.speed = speed
    }

    get_direction_vector() {
        const angle = this.heading * Math.PI / 180
        return [Math.sin(angle), Math.cos(angle)]
    }

    move(seconds = 1) {
        const e = this.get_direction_vector()
        this.x += e[0] * this.speed * seconds / 3600
        this.y += e[1] * this.speed * seconds / 3600
    }

    get_moved_copy(seconds) {
        let pos = new Position(this.x, this.y, this.heading, this.speed)
        pos.move(seconds)
        return pos
    }

    create_position(rel_direction, distance, heading, speed) {
        const angle = (90 - (this.heading + rel_direction)) * Math.PI / 180
        return new Position(
            this.x + distance * Math.cos(angle),
            this.y + distance * Math.sin(angle),
            heading,
            speed)
    }

    get_relative_position(pos) {
        const dx = pos.x - this.x
        const dy = pos.y - this.y
        const distance = Math.sqrt(dx * dx + dy * dy)
        const angle = 90 - (Math.atan2(dy, dx) * 180 / Math.PI)
        const rel_direction = (angle - this.heading + 720) % 360
        return {distance, rel_direction}
    }

    to_svg_path(size = 0.4) {
        const efwd_x = Math.sin(this.heading * Math.PI / 180)
        const efwd_y = Math.cos(this.heading * Math.PI / 180)
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
class Ship {
    static default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    constructor(id = 0, ship_type = "Unbekannt", name = "Kontakt", tons = 0, pos = null, color = null, alive = true) {
        if (pos == null) {
            pos = new Position(0, 0, 0, 0)
        }
        if (color == null) {
            this.color = Ship.default_colors[0]
        } else {
            this.color = color
        }
        this.id = id
        this.alive = alive
        this.ship_type = ship_type
        this.name = name
        this.tons = tons
        this.pos = pos
    }

    is_warship(){
        return this.ship_type == "Zerstörer"
    }
    is_unknown(){
        return this.ship_type == "Unbekannt"
    }
}

function increment_ship_id(state) {
    state.ship_id_counter++
    return state.ship_id_counter
}

function add_ship(state, ship_data) {
    const color = Ship.default_colors.find(c => {
        const ship = state.ships.find(s => s.color == c);
        return ship == null
    })
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
        pos,
        color
    )
    state.ships.push(ship)
}

function pad(num, size) {
    num = num.toString();
    while (num.length < size) num = "0" + num;
    return num;
}

export function seconds_to_timestamp(time) {
    while(time > 86400){
        time -= 86400
    }
    const h = pad(Math.floor(time / 3600), 2)
    const m = pad(Math.floor(((time - 3600 * h) / 60)), 2)
    const s = pad(Math.floor((time - 3600 * h - 60 * m)), 2)
    return h + ":" + m + ":" + s
}

export function timestamp_to_seconds(t) {
    const h = parseFloat(t.substring(0, 2))
    const m = parseFloat(t.substring(3, 5))
    const s = parseFloat(t.substring(6, 8))
    return 3600 * h + 60 * m + s
}

const store = new Vuex.Store({
    state: {
        ship_id_counter: 0,
        time: 0,
        play_speed: 1,
        uboot_pos: new Position(0, 0, 0, 8),
        ships: [],
        intercept: null
    },
    mutations: {
        add_ship: add_ship,
        sink_ship(state, id) {
            state.ships.find(s => s.id == id).alive = false
        },
        remove_ship(state, id) {
            state.ships = state.ships.filter(s => s.id != id)
        },
        set_play_speed(state, speed) {
            state.play_speed = speed
        },
        pass_time: function (state, seconds = 1) {
            state.time += seconds
            while (state.time < 0) {
                state.time += 86400
            }
            while (state.time > 86400) {
                state.time -= 86400
            }
            state.ships.map((s) => s.pos.move(seconds))
            state.uboot_pos.move(seconds)
        },
        set_time(state, time){
            state.time = time
        },
        update_uboot_heading(state, heading) {
            state.uboot_pos.heading = heading
        },
        update_uboot_speed(state, speed) {
            state.uboot_pos.speed = speed
        },
        update_ship_speed(state, {id, speed}) {
            state.ships.filter((ship) => ship.id == id).map(ship => ship.pos.speed = speed)
        },
        update_ship_heading(state, {id, heading}) {
            state.ships.filter((ship) => ship.id == id).map(ship => ship.pos.heading = heading)
        },
        update_ship_distance(state, {id, dist}) {
            state.ships.filter((ship) => ship.id == id).map(ship => {
                const rel_dir = state.uboot_pos.get_relative_position(ship.pos).rel_direction
                ship.pos = state.uboot_pos.create_position(rel_dir, dist, ship.pos.heading, ship.pos.speed)
            })
        },
        update_ship_rel_direction(state, {id, rel_direction}) {
            state.ships.filter(ship => ship.id == id).map(ship => {
                const distance = state.uboot_pos.get_relative_position(ship.pos).distance
                ship.pos = state.uboot_pos.create_position(rel_direction, distance, ship.pos.heading, ship.pos.speed)
            })
        },
        set_intercept(state, intercept) {
            state.intercept = intercept
            const final_wp = intercept.route.at(-1)
            state.intercept.final_uboot_pos = new Position(final_wp.x, final_wp.y, final_wp.heading, final_wp.speed)
            state.intercept.final_ship_positions = state.ships.map(s => {
                return {pos: s.pos.get_moved_copy(final_wp.duration), color: s.color, name: s.name}
            })
            state.intercept.route.map(wp => {wp.timestamp = seconds_to_timestamp(wp.t)})
        }
    }
})

setInterval(() => store.commit('pass_time', store.state.play_speed), 1000)

new Vue({
    render: h => h(App),
    store: store
}).$mount('#app')
