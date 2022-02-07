<template>
  <b-card class="my-1 cardbackground">
    <form v-on:submit.prevent="get_intercept_course">
      <b-form-row>
        <b-col>
          <b-form-group label-for="target_ship" label="Ziel">
            <b-form-select v-model="target_id" id="target_ship" :options="target_options"/>
          </b-form-group>
        </b-col>
        <b-col cols="4">
          <b-form-group label="Seite" label-for="attack_side">
            <b-form-select id="attack_side" v-model="attack_side"
                           :options="[{value:0, text:'Beliebig'},{value:1, text:'Links'}, {value:2, text:'Rechts'}]"/>
          </b-form-group>
        </b-col>
      </b-form-row>
      <b-form-row>
        <b-col>
          <b-form-group label="Segm." label-for="n_segments">
            <b-form-input id="n_segments" v-model="n_segments" type="number" step="1" max="5" min="1"/>
          </b-form-group>
        </b-col>
        <b-col>
          <b-form-group label-for="desired_duration" label="Zeit / h">
            <b-form-input v-model="desired_duration" id="desired_duration" type="number" step="0.1" min="0"/>
          </b-form-group>
        </b-col>
        <b-col>
          <b-form-group label-for="desired_distance" label="Dist.">
            <b-form-input v-model="desired_distance" id="desired_distance" type="number" step="0.1" min="0"/>
          </b-form-group>
        </b-col>
        <b-col>
          <b-form-group label-for="attack_bearing" label="Peilung">
            <b-form-input v-model="attack_bearing" id="attack_bearing" type="number" step="1"/>
          </b-form-group>
        </b-col>
      </b-form-row>
      <b-form-row>
        <b-col class="my-auto">
          <b-form-checkbox v-model="fix_initial_angle" name="check-button" switch class="my-1">1. Kurs fix</b-form-checkbox>
        </b-col>
        <b-col class="col-md-auto">
          <b-button type="submit" variant="danger">Abfangen</b-button>
        </b-col>
      </b-form-row>
    </form>
    <div v-if="intercept != null">
      <b-table :items="intercept_table_data" class="mt-2"/>
    </div>
  </b-card>
</template>


<script>
import axios from 'axios';

export default {
  name: "InterceptControl",
  data() {
    return {
      n_segments: 2,
      target_id: 0,
      // backend_url: "http://localhost:5000/api",
      backend_url: "https://uboot.scherbela.com/api",
      desired_duration: 0,
      desired_distance: 1.0,
      fix_initial_angle: true,
      attack_side: 0,
      attack_bearing: 0
    }
  },
  props: {
    ships: Array,
    uboot_pos: Object,
    time: Number,
    intercept: Object
  },
  computed: {
    target_options() {
      return this.ships.map((s) => {
        return {value: s.id, text: s.name}
      })
    },
    intercept_table_data() {
      if (this.intercept == null) {
        return null
      }
      return this.intercept.route.map(wp => {
        return {
          Uhrzeit: wp.timestamp,
          Kurs: wp.new_heading.toFixed(1),
          Distanz: wp.target_dist.toFixed(1),
          Peilung: wp.target_bearing.toFixed(0)
        }
      })
    }
  },
  methods: {
    get_intercept_params() {
      const target = this.ships.find(s => s.id == this.target_id).pos
      return {
        x: this.uboot_pos.x,
        y: this.uboot_pos.y,
        heading: this.uboot_pos.heading,
        speed: this.uboot_pos.speed,
        target_x: target.x,
        target_y: target.y,
        target_heading: target.heading,
        target_speed: target.speed,
        time: this.time,
        n_segments: this.n_segments,
        desired_duration: this.desired_duration * 3600,
        desired_distance: this.desired_distance,
        fix_initial_angle: this.fix_initial_angle,
        attack_bearing: this.attack_bearing,
        attack_position_angle: this.attack_side == 1 ? -90 : 90,
        fix_attack_side: this.attack_side != 0
      }
    },
    get_intercept_course() {
      axios.get(this.backend_url,
          {params: this.get_intercept_params()}
      ).then(response => {
            console.log(response.data);
            this.$store.commit('set_intercept', response.data)
          }
      ).catch(e => console.log(e))
    }
  }
}
</script>

<style scoped>

label {
  font-size: 8pt;
}

</style>