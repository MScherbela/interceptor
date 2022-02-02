<template>
  <div class="ship">
    <div class="card" :class="{sunk: !ship.alive}">
      <b-card-header class="pt-2 pb-2" :style="{'background-color': ship.color, color: 'white'}">
        <b-row>
          <b-col class="my-auto">
            <b-card-title class="mb-0">
              {{ ship.name }}: {{ ship.tons }} BRT
            </b-card-title>
          </b-col>
          <b-col class="col-md-auto">
            <b-icon-exclamation-circle-fill font-scale="1.5" v-if="ship.is_warship()"/>
            <b-icon-question-circle-fill font-scale="1.5" v-if="ship.is_unknown()"/>
          </b-col>
          <b-col class="col-md-auto">
            <b-button-close v-on:click="$store.commit('remove_ship', id)"/>
          </b-col>
        </b-row>
      </b-card-header>

      <b-card-body>
        <form>
          <b-form-row>
            <b-col>
              <b-form-group label="Distanz" label-for="ship_distance">
                <b-form-input v-model="distance" id="ship_distance" type="number" step="0.01"/>
              </b-form-group>
            </b-col>
            <b-col>
              <b-form-group label="Peilung" label-for="ship_rel_direction">
                <b-form-input v-model="rel_direction" id="ship_rel_direction" type="number"/>
              </b-form-group>
            </b-col>
            <b-col>
              <b-form-group label="Kurs" label-for="ship_heading">
                <b-form-input v-model="heading" id="ship_heading" type="number"/>
              </b-form-group>
            </b-col>

            <b-col>
              <b-form-group label="kn" label-for="ship_speed">
                <b-form-input v-model="speed" id="ship_speed" type="number"/>
              </b-form-group>
            </b-col>
          </b-form-row>
        </form>
      </b-card-body>
    </div>
  </div>
</template>

<script>


export default {
  name: "ShipCard",
  props: {
    id: Number,
    ship: Object
  },
  computed: {
    distance: {
      get() {
        return this.$store.state.uboot_pos.get_relative_position(this.ship.pos).distance.toFixed(1)
      },
      set(dist) {
        this.$store.commit('update_ship_distance', {id: this.ship.id, dist: parseFloat(dist) || 1})
      }
    },
    heading: {
      get() {
        return this.ship.pos.heading
      },
      set(heading) {
        this.$store.commit('update_ship_heading', {id: this.ship.id, heading: parseFloat(heading) || 0})
      }
    },
    rel_direction: {
      get() {
        return this.$store.state.uboot_pos.get_relative_position(this.ship.pos).rel_direction.toFixed(0)
      },
      set(rel_dir) {
        this.$store.commit('update_ship_rel_direction', {id: this.ship.id, rel_direction: parseFloat(rel_dir) || 0})
      }
    },
    speed: {
      get() {
        return this.ship.pos.speed
      },
      set(speed) {
        this.$store.commit('update_ship_speed', {id: this.ship.id, speed: parseFloat(speed) || 0})
      }
    },
  }
}
</script>

<style scoped>
.ship {
  margin-top: 20px;
  margin-bottom: 20px;
}

.sunk {
  opacity: 0.4;
}

.card-title {
  font-size: 12pt;
  font-weight: bold;
}

.card-body {
  padding: 10px;
  padding-bottom: 0px;
}

input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  margin: 0;
}

.form-control {
  padding-right: 5px;
  padding-left: 5px;
}

</style>