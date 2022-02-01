<template>
  <div id="app" data="data" class="fill">
    <b-container fluid style="width: 95%" class="mt-2 mb-2">
      <b-row class="vh-100">
        <b-col  xl="6" md="12" cols="12" class="flex-column d-flex">
          <b-row><b-col><h3>Angriffskarte</h3></b-col></b-row>
          <b-row class="flex-grow-1" style="min-height: 400px">
            <b-col>
              <ShipMap :ships="$store.state.ships" :uboot_pos="$store.state.uboot_pos" :intercept="$store.state.intercept"/>
            </b-col>
          </b-row>
        </b-col>
        <b-col xl="3" md="6" cols="12">
          <b-row><b-col><h3>Ziele</h3></b-col></b-row>
          <b-row>
            <b-col>
              <ShipList/>
            </b-col>
          </b-row>
        </b-col>
        <b-col xl="3" md="6" cols="12">
          <b-row>
            <b-col><h3>Einstellungen</h3></b-col></b-row>
          <b-row>
            <b-col>
              <TimeControl/>
              <UBootInput/>
              <InterceptControl
                  :ships="$store.state.ships"
                  :uboot_pos="$store.state.uboot_pos"
                  :time="this.$store.state.time"
                  :intercept="$store.state.intercept"
              />
            </b-col>
          </b-row>
        </b-col>
      </b-row>
    </b-container>
  </div>
</template>

<script>
import ShipList from './components/ShipLIst.vue'
import ShipMap from "./components/ShipMap";
import UBootInput from "@/components/UBootInput";
import TimeControl from "@/components/TimeControl";
import InterceptControl from "@/components/InterceptControl";

export default {
  name: 'App',
  components: {
    TimeControl,
    ShipList,
    ShipMap,
    UBootInput,
    InterceptControl
  },

}
</script>

<style>
.fill {
  min-height: 100%;
  height: 100%;
}
/*#app {*/
/*  font-family: Avenir, Helvetica, Arial, sans-serif;*/
/*  -webkit-font-smoothing: antialiased;*/
/*  -moz-osx-font-smoothing: grayscale;*/
/*  text-align: center;*/
/*  color: #2c3e50;*/
/*  margin-top: 60px;*/
/*}*/
</style>
